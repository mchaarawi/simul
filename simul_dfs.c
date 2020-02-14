/*
 *  Copyright (C) 2003, The Regents of the University of California.
 *  Produced at the Lawrence Livermore National Laboratory.
 *  Written by Christopher J. Morrone <morrone@llnl.gov>
 *  UCRL-CODE-2003-019
 *  All rights reserved.
 *
 *  Please read the COPYING file.
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License (as published by
 *  the Free Software Foundation) version 2, dated June 1991.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  terms and conditions of the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Copyright (C) 2020 Intel Corporation
 */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <errno.h>
#include <time.h>
#include <sys/time.h>

#include <daos.h>
#include <daos_fs.h>
#include <gurt/common.h>

struct daos_hdls {
	uuid_t		puuid;
	uuid_t		cuuid;
	daos_handle_t	poh;
	daos_handle_t	coh;
	dfs_t		*dfs;
};

#define FILEMODE S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH
#define DIRMODE S_IRUSR|S_IXUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IXOTH
#define SHARED 1
#define MAX_FILENAME_LEN 512

int rank;
int size;
char *testdir = NULL;
char hostname[1024];
int verbose;
int throttle = 1;
struct timeval t1, t2;
static char version[] = "1.16";
static struct daos_hdls hdl;

static void
shutdown_daos()
{
    dfs_umount(hdl.dfs);
    MPI_Barrier(MPI_COMM_WORLD);
    daos_cont_close(hdl.coh, NULL);
    MPI_Barrier(MPI_COMM_WORLD);
    daos_pool_disconnect(hdl.poh, NULL);
    MPI_Barrier(MPI_COMM_WORLD);
    daos_fini();
    MPI_Barrier(MPI_COMM_WORLD);
}

/* For DAOS methods. */
#define DCHECK(rc, format, ...)						\
	do {								\
	    int _rc = (rc);						\
									\
	    if (_rc != 0) {						\
		    fprintf(stderr, "ERROR (%s:%d): %d: %d: "		\
			    format"\n", __FILE__, __LINE__, rank, _rc,	\
			    ##__VA_ARGS__);				\
		    fflush(stderr);					\
		    shutdown_daos();					\
		    MPI_Abort(MPI_COMM_WORLD, -1);			\
	    }								\
	} while (0)

#ifdef __GNUC__
   /* "inline" is a keyword in GNU C */
#elif __STDC_VERSION__ >= 199901L
   /* "inline" is a keyword in C99 and later versions */
#else
#  define inline /* "inline" not available */
#endif

#ifndef AIX
#define FAIL(msg) do { \
    fprintf(stdout, "Process %d: FAILED in %s LINE %d, %s: %s\n",\
	    rank, __func__, __LINE__, msg, strerror(rc));	 \
    fflush(stdout);		   \
    shutdown_daos();		   \
    MPI_Abort(MPI_COMM_WORLD, 1);  \
} while(0)
#else
#define FAIL(msg) do { \
    fprintf(stdout, "%s: Process %d(%s): FAILED, %s: %s\n",\
	    timestamp(), rank, hostname, \
	    msg, strerror(errno)); \
    fflush(stdout);\
    MPI_Abort(MPI_COMM_WORLD, 1); \
} while(0)
#endif

char *timestamp() {
    static char datestring[80];
    time_t timestamp;

    fflush(stdout);
    timestamp = time(NULL);
    strftime(datestring, 80, "%T", localtime(&timestamp));

    return datestring;
}

inline void begin(char *str) {
    if (verbose > 0 && rank == 0) {
        gettimeofday(&t1, NULL);
        fprintf(stdout, "%s:\tBeginning %s\n", timestamp(), str);
        fflush(stdout);
    }
}

inline void end(char *str) {
    double elapsed;

    MPI_Barrier(MPI_COMM_WORLD);
    if (verbose > 0 && rank == 0) {
	gettimeofday(&t2, NULL);
	elapsed = ((((t2.tv_sec - t1.tv_sec) * 1000000L)
		    + t2.tv_usec) - t1.tv_usec)
 		  / (double)1000000;
	if (elapsed >= 60) {
	    fprintf(stdout, "%s:\tFinished %-15s(%.2f min)\n",
		    timestamp(), str, elapsed / 60);
	} else {
	    fprintf(stdout, "%s:\tFinished %-15s(%.3f sec)\n",
		    timestamp(), str, elapsed);

	}
	fflush(stdout);
    }
}

void Seq_begin(MPI_Comm comm, int numprocs) {
    int size;
    int rank;
    int buf;
    MPI_Status status;

    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    if (rank >= numprocs) {
	MPI_Recv(&buf, 1, MPI_INT, rank-numprocs, 1333, comm, &status);
    }
}

void Seq_end(MPI_Comm comm, int numprocs) {
    int size;
    int rank;
    int buf;

    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    if ((rank + numprocs) < size) {
	MPI_Send(&buf, 1, MPI_INT, rank+numprocs, 1333, comm);
    }
}

/* This function does not FAIL if the requested "name" does not exist.  This
   is just to clean up any files or directories left over from previous runs.*/
void remove_file_or_dir(char *name) {
    dfs_remove(hdl.dfs, NULL, name, 1, NULL);
}

char *create_files(char *prefix, int filesize, int shared) {
    static char filename[MAX_FILENAME_LEN];
    char errmsg[MAX_FILENAME_LEN+20];
    int i, rc;
    dfs_obj_t *file;

    /* Process 0 creates the test file(s) */
    if (rank == 0) {
	for (i = 0; i < (shared ? 1 : size); i++) {
            mode_t mode = FILEMODE;

	    sprintf(filename, "%s.%d", prefix, i);
	    remove_file_or_dir(filename);

	    mode = S_IFREG | mode;
	    rc = dfs_open(hdl.dfs, NULL, filename, mode, O_CREAT | O_RDWR,
			  0, 0, NULL, &file);
	    if (rc) {
		sprintf(errmsg, "creat of file %s", filename);
		FAIL(errmsg);
	    }

	    if (filesize > 0) {
		rc = dfs_punch(hdl.dfs, file, filesize, 0);
		if (rc) {
		    sprintf(errmsg, "dfs_set_size in file %s", filename);
		    FAIL(errmsg);
		}
	    }

	    if (dfs_release(file)) {
		sprintf(errmsg, "close of file %s", filename);
		FAIL(errmsg);
	    }
	}
    }

    if (shared)
	sprintf(filename, "%s.0", prefix);
    else
	sprintf(filename, "%s.%d", prefix, rank);

    return filename;
}

void remove_files(char *prefix, int shared) {
    char filename[1024];
    int i, rc;

    /* Process 0 removes the file(s) */
    if (rank == 0) {
	for (i = 0; i < (shared ? 1 : size); i++) {
	    sprintf(filename, "%s.%d", prefix, i);
	    /*printf("Removing file %s\n", filename); fflush(stdout);*/
	    rc = dfs_remove(hdl.dfs, NULL, filename, 1, NULL);
	    if (rc) {
		FAIL("unlink failed");
	    }
	}
    }
}

char *create_dirs(char *prefix, int shared) {
    static char dirname[1024];
    int i, rc;

    /* Process 0 creates the test file(s) */
    if (rank == 0) {
	for (i = 0; i < (shared ? 1 : size); i++) {
	    sprintf(dirname, "%s.%d", prefix, i);
	    remove_file_or_dir(dirname);
	    rc = dfs_mkdir(hdl.dfs, NULL, dirname, DIRMODE, OC_S1);
	    if (rc) {
		FAIL("init mkdir failed");
	    }
	}
    }

    if (shared)
	sprintf(dirname, "%s.0", prefix);
    else
	sprintf(dirname, "%s.%d", prefix, rank);

    return dirname;
}

void remove_dirs(char *prefix, int shared) {
    char dirname[1024];
    int i, rc;

    /* Process 0 removes the file(s) */
    if (rank == 0) {
	for (i = 0; i < (shared ? 1 : size); i++) {
	    sprintf(dirname, "%s.%d", prefix, i);
	    rc = dfs_remove(hdl.dfs, NULL, dirname, 1, NULL);
	    if (rc) {
		FAIL("rmdir failed");
	    }
	}
    }
}

char *create_symlinks(char *prefix, int shared) {
    static char filename[1024];
    static char linkname[1024];
    int i, rc;

    /* Process 0 creates the test file(s) */
    if (rank == 0) {
	for (i = 0; i < (shared ? 1 : size); i++) {
            dfs_obj_t *sym;

	    sprintf(filename, "symlink_target");
	    sprintf(linkname, "%s.%d", prefix, i);
	    remove_file_or_dir(linkname);
	    rc = dfs_open(hdl.dfs, NULL, linkname, S_IFLNK, O_CREAT,
			  0, 0, filename, &sym);
	    if (rc) {
		FAIL("symlink failed");
	    }

	    dfs_release(sym);
	}
    }

    if (shared)
	sprintf(linkname, "%s.0", prefix);
    else
	sprintf(linkname, "%s.%d", prefix, rank);

    return linkname;
}

void check_single_success(char *testname, int ret, int error_rc) {
    int *rc_vec, i;
    int fail = 0;
    int pass = 0;
    int rc;

    if (rank == 0) {
	if ((rc_vec = (int *)malloc(sizeof(int)*size)) == NULL) {
	    rc = ENOMEM;
	    FAIL("malloc failed");
	}
    }
    MPI_Gather(&ret, 1, MPI_INT, rc_vec, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
	for (i = 0; i < size; i++) {
	    if (rc_vec[i] != 0)
		fail++;
	    else
		pass++;
	}
	if (!((pass == 1) && (fail == size-1))) {
	    fprintf(stdout, "%s: FAILED in %s: ", timestamp(), testname);
	    if (pass > 1)
		fprintf(stdout, "too many operations succeeded (%d).\n", pass);
	    else
		fprintf(stdout, "too many operations failed (%d).\n", fail);
	    fflush(stdout);
	    MPI_Abort(MPI_COMM_WORLD, 1);
	}
	free(rc_vec);
    }
}

void simul_open(int shared) {
    char *filename;
    dfs_obj_t *file;
    int rc;

    begin("setup");
    filename = create_files("simul_open", 0, shared);
    end("setup");

    /* All open the file simultaneously */
    begin("test");
    mode_t mode = 0;
    mode = S_IFREG | mode;
    rc = dfs_open(hdl.dfs, NULL, filename, mode, O_RDWR,
		  0, 0, NULL, &file);
    if (rc) {
	FAIL("open failed");
    }
    end("test");

    /* All close the file one at a time */
    begin("cleanup");
    Seq_begin(MPI_COMM_WORLD, throttle);
    if (dfs_release(file)) {
	FAIL("close failed");
    }
    Seq_end(MPI_COMM_WORLD, throttle);
    MPI_Barrier(MPI_COMM_WORLD);
    remove_files("simul_open", shared);
    end("cleanup");
}

void simul_close(int shared) {
    char *filename;
    dfs_obj_t *file;
    int rc;

    begin("setup");
    filename = create_files("simul_close", 0, shared);
    MPI_Barrier(MPI_COMM_WORLD);
    /* All open the file one at a time */
    Seq_begin(MPI_COMM_WORLD, throttle);
    mode_t mode = 0;
    mode = S_IFREG | mode;
    rc = dfs_open(hdl.dfs, NULL, filename, mode, O_RDWR,
		  0, 0, NULL, &file);
    if (rc) {
	FAIL("open failed");
    }
    Seq_end(MPI_COMM_WORLD, throttle);
    end("setup");

    begin("test");
    /* All close the file simultaneously */
    if (dfs_release(file)) {
	FAIL("close failed");
    }
    end("test");

    begin("cleanup");
    remove_files("simul_close", shared);
    end("cleanup");
}

void simul_chdir(int shared) {
    printf("skip simul_chdir\n");
    return;
#if 0
    char cwd[1024];
    char *dirname;

    begin("setup");
    if (getcwd(cwd, 1024) == NULL) {
	FAIL("init getcwd failed");
    }
    dirname = create_dirs("simul_chdir", shared);
    end("setup");

    begin("test");
    /* All chdir to dirname */
    if (chdir(dirname) == -1) {
	FAIL("chdir failed");
    }
    end("test");

    begin("cleanup");
    /* All chdir back to old cwd */
    if (chdir(cwd) == -1) {
	FAIL("chdir back failed");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    remove_dirs("simul_chdir", shared);
    end("cleanup");
#endif
}

void simul_file_stat(int shared) {
    char *filename;
    struct stat buf;
    int rc;

    begin("setup");
    filename = create_files("simul_file_stat", 0, shared);
    end("setup");

    begin("test");
    /* All stat the file */
    rc = dfs_stat(hdl.dfs, NULL, filename, &buf);
    if (rc) {
	FAIL("stat failed");
    }
    end("test");

    begin("cleanup");
    remove_files("simul_file_stat", shared);
    end("cleanup");
}

void simul_dir_stat(int shared) {
    char *dirname;
    struct stat buf;
    int rc;

    begin("setup");
    dirname = create_dirs("simul_dir_stat", shared);
    end("setup");

    begin("test");
    /* All stat the directory */
    rc = dfs_stat(hdl.dfs, NULL, dirname, &buf);
    if (rc) {
	FAIL("stat failed");
    }
    end("test");

    begin("cleanup");
    remove_dirs("simul_dir_stat", shared);
    end("cleanup");
}

void simul_readdir(int shared) {
    char *dirname;
    dfs_obj_t *dir;
    struct dirent dptr;
    daos_anchor_t anchor = {0};
    unsigned nr;
    int rc;

    begin("setup");
    dirname = create_dirs("simul_readdir", shared);
    MPI_Barrier(MPI_COMM_WORLD);
    /* All open the directory(ies) one at a time */
    Seq_begin(MPI_COMM_WORLD, throttle);
    //mode_t mode = DIRMODE | S_IFDIR;
    //rc = dfs_open(hdl.dfs, NULL, dirname, mode, O_RDWR, 0, 0, NULL, &dir);
    rc = dfs_lookup_rel(hdl.dfs, NULL, dirname, O_RDONLY, &dir, NULL, NULL);
    if (rc) {
	FAIL("init opendir failed");
    }
    Seq_end(MPI_COMM_WORLD, throttle);
    end("setup");

    begin("test");
    /* All readdir the directory stream(s) */
    nr = 1;
    rc = dfs_readdir(hdl.dfs, dir, &anchor, &nr, &dptr);
    if (rc) {
	FAIL("readdir failed");
    }
    end("test");

    begin("cleanup");
    /* All close the directory(ies) one at a time */
    Seq_begin(MPI_COMM_WORLD, throttle);
    if (dfs_release(dir)) {
	FAIL("closedir failed");
    }
    Seq_end(MPI_COMM_WORLD, throttle);
    MPI_Barrier(MPI_COMM_WORLD);
    remove_dirs("simul_readdir", shared);
    end("cleanup");
}

void simul_statfs(int shared) {
    printf("skip statfs\n");
    return;

#if 0
    char *filename;
    struct statfs buf;

    begin("setup");
    filename = create_files("simul_statfs", 0, shared);
    end("setup");

    begin("test");
    /* All statfs the file(s) */
    if (statfs(filename, &buf) == -1) {
	FAIL("statfs failed");
    }
    end("test");

    begin("cleanup");
    remove_files("simul_statfs", shared);
    end("cleanup");
#endif
}

void simul_lseek(int shared) {
    printf("skip lseek\n");
    return;

#if 0
    int fd;
    char *filename;

    begin("setup");
    filename = create_files("simul_lseek", 0, shared);
    MPI_Barrier(MPI_COMM_WORLD);
    /* All open the file(s) one at a time */
    Seq_begin(MPI_COMM_WORLD, throttle);
    if ((fd = open(filename, O_RDWR)) == -1) {
	FAIL("init open failed");
    }
    Seq_end(MPI_COMM_WORLD, throttle);
    end("setup");

    begin("test");
    /* All lseek simultaneously */
    if (lseek(fd, 1024, SEEK_SET) == -1) {
	FAIL("lseek failed");
	MPI_Abort(MPI_COMM_WORLD, 1);
    }
    end("test");

    begin("cleanup");
    /* All close the file(s) one at a time */
    Seq_begin(MPI_COMM_WORLD, throttle);
    if (close(fd) == -1) {
	FAIL("cleanup close failed");
    }
    Seq_end(MPI_COMM_WORLD, throttle);
    MPI_Barrier(MPI_COMM_WORLD);
    remove_files("simul_lseek", shared);
    end("cleanup");
#endif
}

void simul_read(int shared) {
    daos_size_t fin;
    off_t offset = 0;
    char buf[1024];
    char *filename;
    int i = 0;
    int retry = 100;
    dfs_obj_t *file;
    int rc;

    begin("setup");
    filename = create_files("simul_read", 1024, shared);
    MPI_Barrier(MPI_COMM_WORLD);
    /* All open the file one at a time */
    Seq_begin(MPI_COMM_WORLD, throttle);
    rc = dfs_open(hdl.dfs, NULL, filename, S_IFREG, O_RDWR, 0, 0, NULL, &file);
    if (rc) {
	FAIL("init open failed");
    }
    Seq_end(MPI_COMM_WORLD, throttle);
    end("setup");

    begin("test");
    /* All read simultaneously */
    for (i = 1024; (i > 0) && (retry > 0); i -= fin, retry--) {
	d_sg_list_t sgl;
	d_iov_t iov;

	sgl.sg_nr = 1;
	sgl.sg_nr_out = 0;
	d_iov_set(&iov, buf, (size_t)i);
	sgl.sg_iovs = &iov;

	rc = dfs_read(hdl.dfs, file, &sgl, offset, &fin, NULL);
	if (rc) {
		FAIL("read failed");
	}
	offset += fin;
    }
    if( (retry == 0) && (i > 0) )
	FAIL("read exceeded retry count");
    end("test");

    begin("cleanup");
    /* All close the file one at a time */
    Seq_begin(MPI_COMM_WORLD, throttle);
    if (dfs_release(file)) {
	FAIL("cleanup close failed");
    }
    Seq_end(MPI_COMM_WORLD, throttle);
    MPI_Barrier(MPI_COMM_WORLD);
    remove_files("simul_read", shared);
    end("cleanup");
}

void simul_write(int shared) {
    off_t offset = 0;
    char *filename;
    int i = 0;
    dfs_obj_t *file;
    int rc;

    begin("setup");
    filename = create_files("simul_write", size * sizeof(int), shared);
    MPI_Barrier(MPI_COMM_WORLD);
    /* All open the file and lseek one at a time */
    Seq_begin(MPI_COMM_WORLD, throttle);
    rc = dfs_open(hdl.dfs, NULL, filename, S_IFREG, O_RDWR, 0, 0, NULL, &file);
    if (rc) {
	FAIL("init open failed");
    }
    offset = rank*sizeof(int);
    Seq_end(MPI_COMM_WORLD, throttle);
    end("setup");

    begin("test");
    /* All write simultaneously */
    i = sizeof(int);

    d_sg_list_t sgl;
    d_iov_t iov;

    sgl.sg_nr = 1;
    sgl.sg_nr_out = 0;
    d_iov_set(&iov, &rank, (size_t)i);
    sgl.sg_iovs = &iov;

    rc = dfs_write(hdl.dfs, file, &sgl, offset, NULL);
    if (rc) {
	    FAIL("write failed");
    }
    end("test");

    begin("cleanup");
    /* All close the file one at a time */
    Seq_begin(MPI_COMM_WORLD, throttle);
    if (dfs_release(file)) {
	FAIL("cleanup close failed");
    }
    Seq_end(MPI_COMM_WORLD, throttle);
    MPI_Barrier(MPI_COMM_WORLD);
    remove_files("simul_write", shared);
    end("cleanup");
}

void simul_mkdir(int shared) {
    int rc, i;
    char dirname[MAX_FILENAME_LEN];

    begin("setup");
    if (shared)
	sprintf(dirname, "simul_mkdir.0");
    else
	sprintf(dirname, "simul_mkdir.%d", rank);
    if (rank == 0) {
	for (i = 0; i < (shared ? 1 : size); i++) {
	    char buf[MAX_FILENAME_LEN];
	    sprintf(buf, "simul_mkdir.%d", i);
	    remove_file_or_dir(buf);
	}
    }
    end("setup");

    begin("test");
    /* All mkdir dirname */
    rc = dfs_mkdir(hdl.dfs, NULL, dirname, DIRMODE, OC_S1);
    if (!shared) {
        if (rc) {
	    FAIL("mkdir failed");
	}
	MPI_Barrier(MPI_COMM_WORLD);
    } else { /* Only one should succeed */
	check_single_success("simul_mkdir", rc, -1);
    }
    end("test");

    begin("cleanup");
    remove_dirs("simul_mkdir", shared);
    end("cleanup");
}

void simul_rmdir(int shared) {
    int rc;
    char *dirname;

    begin("setup");
    dirname = create_dirs("simul_rmdir", shared);
    MPI_Barrier(MPI_COMM_WORLD);
    end("setup");

    begin("test");
    /* All rmdir dirname */
    rc = dfs_remove(hdl.dfs, NULL, dirname, 1, NULL);
    if (!shared) {
	if (rc) {
	    FAIL("rmdir failed");
	}
	MPI_Barrier(MPI_COMM_WORLD);
    } else { /* Only one should succeed */
	check_single_success("simul_rmdir", rc, -1);
    }
    end("test");

    begin("cleanup");
    end("cleanup");
}

void simul_creat(int shared) {
    int i, rc;
    dfs_obj_t *file;
    char filename[1024];
    mode_t mode = FILEMODE;

    begin("setup");
    if (shared)
	sprintf(filename, "simul_creat.0");
    else
	sprintf(filename, "simul_creat.%d", rank);
    if (rank == 0) {
	for (i = 0; i < (shared ? 1 : size); i++) {
	    char buf[MAX_FILENAME_LEN];
	    sprintf(buf, "simul_creat.%d", i);
	    remove_file_or_dir(buf);
	}
    }
    end("setup");

    begin("test");
    mode = S_IFREG | mode;
    /* All create the files simultaneously */
    rc = dfs_open(hdl.dfs, NULL, filename, mode, O_CREAT | O_RDWR,
		  0, 0, NULL, &file);
    if (rc) {
	FAIL("creat failed");
    }
    rc = dfs_punch(hdl.dfs, file, 0, DFS_MAX_FSIZE);
    if (rc) {
	FAIL("creat failed");
    }
    end("test");

    begin("cleanup");
    /* All close the files one at a time */
    Seq_begin(MPI_COMM_WORLD, throttle);
    if (dfs_release(file)) {
	FAIL("close failed");
    }
    Seq_end(MPI_COMM_WORLD, throttle);
    MPI_Barrier(MPI_COMM_WORLD);
    remove_files("simul_creat", shared);
    end("cleanup");
}

void simul_unlink(int shared) {
    int rc;
    char *filename;

    begin("setup");
    filename = create_files("simul_unlink", 0, shared);
    end("setup");

    begin("test");
    /* All unlink the files simultaneously */
    rc = dfs_remove(hdl.dfs, NULL, filename, 1, NULL);
    if (!shared) {
	if (rc) {
	    FAIL("unlink failed");
	}
    } else {
	check_single_success("simul_unlink", rc, -1);
    }
    end("test");

    begin("cleanup");
    end("cleanup");
}

void simul_rename(int shared) {
    int rc, i;
    char *oldfilename;
    char newfilename[1024];
    char *testname = "simul_rename";

    begin("setup");
    oldfilename = create_files(testname, 0, shared);
    sprintf(newfilename, "%s_new.%d", testname, rank);
    if (rank == 0) {
	for (i = 0; i < (shared ? 1 : size); i++) {
	    char buf[MAX_FILENAME_LEN];
	    sprintf(buf, "%s_new.%d", testname, i);
	    remove_file_or_dir(buf);
	}
    }
    end("setup");

    begin("test");
    /* All rename the files simultaneously */
    rc = dfs_move(hdl.dfs, NULL, oldfilename, NULL, newfilename, NULL);
    if (!shared) {
	if (rc) {
	    FAIL("stat failed");
	}
    } else {
	check_single_success(testname, rc, -1);
    }
    end("test");

    begin("cleanup");
    if (rc == 0) {
        rc = dfs_remove(hdl.dfs, NULL, newfilename, 1, NULL);
	if (rc)
	    FAIL("unlink failed");
    }
    end("cleanup");
}

void simul_truncate(int shared) {
    char *filename;
    dfs_obj_t *file;
    int rc;

    begin("setup");
    filename = create_files("simul_truncate", 2048, shared);
    end("setup");

    begin("test");
    /* All truncate simultaneously */
    rc = dfs_open(hdl.dfs, NULL, filename, S_IFREG, O_RDWR, 0, 0, NULL, &file);
    if (rc) {
	FAIL("dfs_open failed");
    }
    rc = dfs_punch(hdl.dfs, file, 1024, DFS_MAX_FSIZE);
    if (rc) {
	FAIL("truncate failed");
    }
    dfs_release(file);
    end("test");

    begin("cleanup");
    remove_files("simul_truncate", shared);
    end("cleanup");
}

void simul_readlink(int shared) {
    char *linkname;
    char buf[1024];
    dfs_obj_t *sym;
    int rc;

    begin("setup");
    linkname = create_symlinks("simul_readlink", shared);
    end("setup");

    begin("test");
    rc = dfs_lookup_rel(hdl.dfs, NULL, linkname, O_RDONLY, &sym, NULL, NULL);
    if (rc)
	    FAIL("dfs_lookup failed");

    daos_size_t val_size = 1024;

    rc = dfs_get_symlink_value(sym, buf, &val_size);
    if (rc) {
	FAIL("readlink failed");
    }
    end("test");

    begin("cleanup");
    remove_files("simul_readlink", shared);
    end("cleanup");
}

void simul_symlink(int shared) {
    int rc, i;
    char linkname[MAX_FILENAME_LEN];
    char filename[MAX_FILENAME_LEN];
    dfs_obj_t *sym;

    begin("setup");
    if (shared)
	sprintf(linkname, "simul_symlink.0");
    else
	sprintf(linkname, "simul_symlink.%d", rank);
    if (rank == 0) {
	for (i = 0; i < (shared ? 1 : size); i++) {
	    char buf[MAX_FILENAME_LEN];
	    sprintf(buf, "simul_symlink.%d", i);
	    remove_file_or_dir(buf);
	}
    }
    sprintf(filename, "simul_symlink_target");
    end("setup");

    begin("test");
    /* All create the symlinks simultaneously */
    rc = dfs_open(hdl.dfs, NULL, linkname, S_IFLNK, O_CREAT,
		  0, 0, filename, &sym);
    if (rc) {
	    FAIL("symlink failed");
    }
    if (!shared) {
	if (rc) {
	    FAIL("symlink failed");
	}
    } else {
	check_single_success("simul_symlink", rc, -1);
    }
    dfs_release(sym);
    end("test");

    begin("cleanup");
    remove_files("simul_symlink", shared);
    end("cleanup");
}

void simul_link_to_one(int shared) {
    int rc, i;
    char *filename;
    char linkname[1024];

    printf("skip hard link\n");
    return;

    begin("setup");
    if (shared)
	sprintf(linkname, "simul_link.0");
    else
	sprintf(linkname, "simul_link.%d", rank);
    if (rank == 0) {
	for (i = 0; i < (shared ? 1 : size); i++) {
	    char buf[MAX_FILENAME_LEN];
	    sprintf(buf, "simul_link.%d", i);
	    remove_file_or_dir(buf);
	}
    }
    filename = create_files("simul_link_target", 0, SHARED);
    end("setup");

    begin("test");
    /* All create the hard links simultaneously */
    rc = link(filename, linkname);
    if (!shared) {
	if (rc) {
	    FAIL("link failed");
	}
    } else {
	check_single_success("simul_link_to_one", rc, -1);
    }
    end("test");

    begin("cleanup");
    remove_files("simul_link_target", SHARED);
    remove_files("simul_link", shared);
    end("cleanup");
}

void simul_link_to_many(int shared) {
    printf("skip hard link\n");
    return;
#if 0
    char *filename;
    char linkname[1024];
    int i;

    if (shared) {
	if (verbose > 0 && rank == 0)
	    printf("%s:\tThis is just a place holder; no test is run here.\n",
		   timestamp());
	return;
    }
    begin("setup");
    filename = create_files("simul_link", 0, shared);
    sprintf(linkname, "simul_link_target.%d", rank);
    if (rank == 0) {
	for (i = 0; i < size; i++) {
	    char buf[MAX_FILENAME_LEN];
	    sprintf(buf, "simul_link_target.%d", i);
	    remove_file_or_dir(buf);
	}
    }
    end("setup");

    begin("test");
    /* All create the hard links simultaneously */
    if (link(filename, linkname) == -1) {
	FAIL("link failed");
    }
    end("test");

    begin("cleanup");
    remove_files("simul_link", shared);
    remove_files("simul_link_target", !SHARED);
    end("cleanup");
#endif
}

void simul_fcntl_lock(int shared) {
    printf("skip flock\n");
    return;
#if 0
    int rc, fd;
    char *filename;
    struct flock sf_lock = {
        .l_type = F_WRLCK,
	.l_whence = SEEK_SET,
	.l_start = 0,
	.l_len = 0
    };
    struct flock sf_unlock = {
        .l_type = F_UNLCK,
	.l_whence = SEEK_SET,
	.l_start = 0,
	.l_len = 0
    };

    begin("setup");
    filename = create_files("simul_fcntl", 0, shared);
    MPI_Barrier(MPI_COMM_WORLD);
    /* All open the file one at a time */
    Seq_begin(MPI_COMM_WORLD, throttle);
    if ((fd = open(filename, O_RDWR)) == -1) {
	FAIL("open failed");
    }
    Seq_end(MPI_COMM_WORLD, throttle);
    end("setup");

    begin("test");
    /* All lock the file(s) simultaneously */
    rc = fcntl(fd, F_SETLK, &sf_lock);
    if (!shared) {
	if (rc) {
            if (errno == ENOSYS) {
                if (rank == 0) {
                    fprintf(stdout, "WARNING: fcntl locking not supported.\n");
                    fflush(stdout);
                }
            } else {
                FAIL("fcntl lock failed");
            }
	}
	MPI_Barrier(MPI_COMM_WORLD);
    } else {
	int saved_errno = errno;
	int *rc_vec, *er_vec, i;
	int fail = 0;
	int pass = 0;
        int nosys = 0;
	if (rank == 0) {
	    if ((rc_vec = (int *)malloc(sizeof(int)*size)) == NULL)
		FAIL("malloc failed");
	    if ((er_vec = (int *)malloc(sizeof(int)*size)) == NULL)
		FAIL("malloc failed");
	}
	MPI_Gather(&rc, 1, MPI_INT, rc_vec, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&saved_errno, 1, MPI_INT, er_vec, 1, MPI_INT, 0,
		   MPI_COMM_WORLD);
	if (rank == 0) {
	    for (i = 0; i < size; i++) {
	        if (rc_vec[i] == -1) {
                    if (er_vec[i] == ENOSYS) {
                        nosys++;
                    } else if (er_vec[i] != EACCES && er_vec[i] != EAGAIN) {
			errno = er_vec[i];
			FAIL("fcntl failed as expected, but with wrong errno");
		    }
		    fail++;
		} else {
		    pass++;
		}
	    }
            if (nosys == size) {
                fprintf(stdout, "WARNING: fcntl locking not supported.\n");
                fflush(stdout);
            } else if (!((pass == 1) && (fail == size-1))) {
		fprintf(stdout,
			"%s: FAILED in simul_fcntl_lock", timestamp());
		if (pass > 1)
		    fprintf(stdout,
			    "too many fcntl locks succeeded (%d).\n", pass);
		else
		    fprintf(stdout,
			    "too many fcntl locks failed (%d).\n", fail);
		fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, 1);
	    }
	    free(rc_vec);
	    free(er_vec);
	}
    }
    end("test");

    begin("cleanup");
    /* All close the file one at a time */
    Seq_begin(MPI_COMM_WORLD, throttle);
    if (!shared || rank == 0) {
	    rc = fcntl(fd, F_SETLK, &sf_unlock);
            if (rc != ENOSYS)
                FAIL("fcntl unlock failed");
    }
    if (close(fd) == -1) {
	FAIL("close failed");
    }
    Seq_end(MPI_COMM_WORLD, throttle);
    MPI_Barrier(MPI_COMM_WORLD);
    remove_files("simul_fcntl", shared);
    end("cleanup");
#endif
}

struct test {
    char *name;
    void (*function) (int);
    int simul; /* Flag designating support for simultaneus mode */
    int indiv; /* Flag designating support for individual mode */
};

static struct test testlist[] = {
    {"open", simul_open},
    {"close", simul_close},
    {"file stat", simul_file_stat},
    {"lseek", simul_lseek},
    {"read", simul_read},
    {"write", simul_write},
    {"chdir", simul_chdir},
    {"directory stat", simul_dir_stat},
    {"statfs", simul_statfs},
    {"readdir", simul_readdir},
    {"mkdir", simul_mkdir},
    {"rmdir", simul_rmdir},
    {"unlink", simul_unlink},
    {"rename", simul_rename},
    {"creat", simul_creat},
    {"truncate", simul_truncate},
    {"symlink", simul_symlink},
    {"readlink", simul_readlink},
    {"link to one file", simul_link_to_one},
    {"link to a file per process", simul_link_to_many},
    {"fcntl locking", simul_fcntl_lock},
    {0}
};

/* Searches an array of ints for one int.  A "-1" must mark the end of the
   array.  */
int int_in_list(int item, int *list) {
    int *ptr;

    if (list == NULL)
	return 0;
    ptr = list;
    while (*ptr != -1) {
	if (*ptr == item)
	    return 1;
	ptr += 1;
    }
    return 0;
}

/* Breaks string of comma-sperated ints into an array of ints */
int *string_split(char *string) {
    char *ptr;
    char *tmp;
    int excl_cnt = 1;
    int *list;
    int i;
    int rc;

    ptr = string;
    while((tmp = strchr(ptr, ','))) {
	ptr = tmp + 1;
	excl_cnt++;
    }

    list = (int *)malloc(sizeof(int) * (excl_cnt + 1));
    if (list == NULL) {
        rc = ENOMEM;
	FAIL("malloc failed");
    }

    tmp = (strtok(string, ", "));
    if (tmp == NULL) {
	    rc = errno;
	    FAIL("strtok failed");
    }
    list[0] = atoi(tmp);
    for (i = 1; i < excl_cnt; i++) {
	tmp = (strtok(NULL, ", "));
	if (tmp == NULL) {
		rc = errno;
		FAIL("strtok failed");
	}
	list[i] = atoi(tmp);
    }
    list[i] = -1;

    return list;
}

void print_help(int testcnt) {
    int i;

    if (rank == 0) {
	printf("simul-%s\n", version);
	printf("Usage: simul [-h] -d <testdir> [-f firsttest] [-l lasttest]\n");
	printf("             [-n #] [-N #] [-i \"4,7,13\"] [-e \"6,22\"] [-s]\n");
	printf("             [-v] [-V #]\n");
	printf("\t-h: prints this help message\n");
	printf("\t-d: the directory in which the tests will run\n");
	printf("\t-f: the number of the first test to run (default: 0)\n");
	printf("\t-l: the number of the last test to run (default: %d)\n",
	       (testcnt*2)-1);
	printf("\t-i: comma-sperated list of tests to include\n");
	printf("\t-e: comma-sperated list of tests to exclude\n");
	printf("\t-s: single-step through every iteration of every test\n");
	printf("\t-n: repeat each test # times\n");
	printf("\t-N: repeat the entire set of tests # times\n");
	printf("\t-v: increases the verbositly level by 1\n");
	printf("\t-V: select a specific verbosity level\n");
	printf("\nThe available tests are:\n");
	for (i = 0; i < testcnt * 2; i++) {
	    printf("\tTest #%d: %s, %s mode.\n", i,
		   testlist[i%testcnt].name,
		   (i < testcnt) ? "shared" : "individual");
	}
    }

    MPI_Initialized(&i);
    if (i) MPI_Finalize();
    exit(0);
}

static int
share_handles(int mpi_rank, MPI_Comm comm)
{
    char uuid_buf[74];
    d_iov_t pool_hdl = { NULL, 0, 0 };
    d_iov_t cont_hdl = { NULL, 0, 0 };
    d_iov_t dfs_hdl = { NULL, 0, 0 };
    char *buf = NULL;
    uint64_t total_size = 0;
    int rc = 0;

    if (mpi_rank == 0) {
        rc = daos_pool_local2global(hdl.poh, &pool_hdl);
        if (rc)
            return rc;
        rc = daos_cont_local2global(hdl.coh, &cont_hdl);
        if (rc)
            return rc;
        rc = dfs_local2global(hdl.dfs, &dfs_hdl);
        if (rc)
            return rc;

        total_size = sizeof(uuid_buf) + pool_hdl.iov_buf_len + cont_hdl.iov_buf_len +
            dfs_hdl.iov_buf_len + sizeof(daos_size_t) * 3;
    }

    /** broadcast size to all peers */
    rc = MPI_Bcast(&total_size, 1, MPI_UINT64_T, 0, comm);
    if (rc != MPI_SUCCESS)
        return -1;

    /** allocate buffers */
    buf = malloc(total_size);
    if (buf == NULL)
        return -1;

    if (mpi_rank == 0) {
        char *ptr = buf;

        uuid_unparse(hdl.puuid, ptr);
        ptr += 37;
        uuid_unparse(hdl.cuuid, ptr);
        ptr += 37;

        *((daos_size_t *) ptr) = pool_hdl.iov_buf_len;
        ptr += sizeof(daos_size_t);
        pool_hdl.iov_buf = ptr;
        pool_hdl.iov_len = pool_hdl.iov_buf_len;
        rc = daos_pool_local2global(hdl.poh, &pool_hdl);
        if (rc)
            goto out;
        ptr += pool_hdl.iov_buf_len;

        *((daos_size_t *) ptr) = cont_hdl.iov_buf_len;
        ptr += sizeof(daos_size_t);
        cont_hdl.iov_buf = ptr;
        cont_hdl.iov_len = cont_hdl.iov_buf_len;
        rc = daos_cont_local2global(hdl.coh, &cont_hdl);
        if (rc)
            goto out;
        ptr += cont_hdl.iov_buf_len;

        *((daos_size_t *) ptr) = dfs_hdl.iov_buf_len;
        ptr += sizeof(daos_size_t);
        dfs_hdl.iov_buf = ptr;
        dfs_hdl.iov_len = dfs_hdl.iov_buf_len;
        rc = dfs_local2global(hdl.dfs, &dfs_hdl);
        if (rc)
            goto out;
        ptr += dfs_hdl.iov_buf_len;

    }

    rc = MPI_Bcast(buf, total_size, MPI_BYTE, 0, comm);
    if (rc != MPI_SUCCESS)
        goto out;

    if (mpi_rank != 0) {
        char *ptr = buf;

        rc = uuid_parse(ptr, hdl.puuid);
        if (rc)
            goto out;
        ptr += 37;

        rc = uuid_parse(ptr, hdl.cuuid);
        if (rc)
            goto out;
        ptr += 37;

        pool_hdl.iov_buf_len = *((daos_size_t *) ptr);
        ptr += sizeof(daos_size_t);
        pool_hdl.iov_buf = ptr;
        pool_hdl.iov_len = pool_hdl.iov_buf_len;
        rc = daos_pool_global2local(pool_hdl, &hdl.poh);
        if (rc)
            goto out;
        ptr += pool_hdl.iov_buf_len;

        cont_hdl.iov_buf_len = *((daos_size_t *) ptr);
        ptr += sizeof(daos_size_t);
        cont_hdl.iov_buf = ptr;
        cont_hdl.iov_len = cont_hdl.iov_buf_len;
        rc = daos_cont_global2local(hdl.poh, cont_hdl, &hdl.coh);
        if (rc)
            goto out;
        ptr += cont_hdl.iov_buf_len;

        dfs_hdl.iov_buf_len = *((daos_size_t *) ptr);
        ptr += sizeof(daos_size_t);
        dfs_hdl.iov_buf = ptr;
        dfs_hdl.iov_len = dfs_hdl.iov_buf_len;
        rc = dfs_global2local(hdl.poh, hdl.coh, O_RDWR, dfs_hdl, &hdl.dfs);
        if (rc)
            goto out;
    }

out:
    free(buf);
    return rc;
}

void
dfs_err_hdlr( MPI_Comm *comm, int *err, ...)
{
	printf("Invoking DFS ERR HDLR\n");
	shutdown_daos();
	return;
}

int main(int argc, char **argv) {
    int testcnt;
    int first;
    int last;
    int i, j, k, c;
    int *excl_list = NULL;
    int *incl_list = NULL;
    int test;
    int singlestep = 0;
    int iterations = 1;
    int set_iterations = 1;
    char linebuf[80];
    MPI_Errhandler newerr;
    int rc;

    /* Check for -h parameter before MPI_Init so the simul binary can be
       called directly, without, for instance, mpirun. */
    for (testcnt = 1; testlist[testcnt].name != 0; testcnt++) continue;
    for (i = 1; i < argc; i++) {
	if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
	    print_help(testcnt);
	}
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm_create_errhandler(dfs_err_hdlr, &newerr);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, newerr);

    rc = daos_init();
    DCHECK(rc, "Failed to initialize daos");

    if (rank == 0) {
	printf("Simul is running with %d process(es)\n", size);
	fflush(stdout);
    }

    first = 0;
    last = testcnt * 2;

    /* Parse command line options */
    while (1) {
	c = getopt(argc, argv, "d:e:f:hi:l:n:N:svV:");
	if (c == -1)
	    break;

	switch (c) {
	case 'd':
	    testdir = optarg;
	    break;
	case 'e':
	    excl_list = string_split(optarg);
	    break;
	case 'f':
	    first = atoi(optarg);
	    if (first >= last) {
		printf("Invalid parameter, firsttest must be <= lasttest\n");
		MPI_Abort(MPI_COMM_WORLD, 2);
	    }
	    break;
	case 'h':
	    print_help(testcnt);
	    break;
	case 'i':
	    incl_list = string_split(optarg);
	    break;
	case 'l':
	    last = atoi(optarg)+1;
	    if (last <= first) {
		printf("Invalid parameter, lasttest must be >= firsttest\n");
		MPI_Abort(MPI_COMM_WORLD, 2);
	    }
	    break;
	case 'n':
	    iterations = atoi(optarg);
	    break;
	case 'N':
	    set_iterations = atoi(optarg);
	    break;
	case 's':
	    singlestep = 1;
	    break;
	case 'v':
	    verbose += 1;
	    break;
	case 'V':
	    verbose = atoi(optarg);
	    break;
	}
    }

    if (testdir == NULL && rank == 0) {
	printf("Please specify a test directory! (\"simul -h\" for help)\n");
	MPI_Abort(MPI_COMM_WORLD, 2);
    }

    if (gethostname(hostname, 1024) == -1) {
	perror("gethostname");
	MPI_Abort(MPI_COMM_WORLD, 2);
    }

    /* If a list of tests was not specified with the -i option, then use
       the first and last number to build a range of included tests. */
    if (incl_list == NULL) {
	incl_list = (int *)malloc(sizeof(int) * (2+last-first));
	for (i = 0; i < last-first; i++) {
	    incl_list[i] = i + first;
	}
	incl_list[i] = -1;
    }

    if (rank == 0) {
	    d_rank_list_t *svc = NULL;
	    daos_pool_info_t pool_info;
	    daos_cont_info_t co_info;
	    char *pool = NULL, *cont = NULL, *svcl = NULL;

	    /** Parse DAOS pool uuid */
	    pool = getenv("DAOS_POOL");
	    if (!pool) {
		    fprintf(stderr, "missing pool uuid - export DAOS_POOL\n");
		    MPI_Abort(MPI_COMM_WORLD, 1);
	    }
	    if (uuid_parse(pool, hdl.puuid) < 0) {
		    fprintf(stderr, "Invalid pool uuid\n");
		    MPI_Abort(MPI_COMM_WORLD, 1);
	    }

	    /** Parse DAOS cont uuid */
	    cont = getenv("DAOS_CONT");
	    if (!cont) {
		    fprintf(stderr, "missing cont uuid - export DAOS_CONT\n");
		    MPI_Abort(MPI_COMM_WORLD, 1);
	    }
	    if (uuid_parse(cont, hdl.cuuid) < 0) {
		    fprintf(stderr, "Invalid cont uuid\n");
		    MPI_Abort(MPI_COMM_WORLD, 1);
	    }

	    /** Parse DAOS pool SVCL */
	    svcl = getenv("DAOS_SVCL");
	    if (!svcl) {
		    fprintf(stderr, "missing pool service rank list - export DAOS_SVCL\n");
		    MPI_Abort(MPI_COMM_WORLD, 1);
	    }
	    svc = daos_rank_list_parse(svcl, ":");
	    if (svc == NULL) {
		    fprintf(stderr, "Invalid pool service rank list\n");
		    MPI_Abort(MPI_COMM_WORLD, 1);
	    }

	    /** Connect to DAOS pool */
	    rc = daos_pool_connect(hdl.puuid, NULL, svc, DAOS_PC_RW,
				   &hdl.poh, &pool_info, NULL);
	    d_rank_list_free(svc);
	    DCHECK(rc, "Failed to connect to pool");

	    rc = daos_cont_open(hdl.poh, hdl.cuuid, DAOS_COO_RW, &hdl.coh,
				&co_info, NULL);
	    /* If NOEXIST we create it */
	    if (rc == -DER_NONEXIST) {
		    rc = dfs_cont_create(hdl.poh, hdl.cuuid, NULL, &hdl.coh, NULL);
		    if (rc)
			    DCHECK(rc, "Failed to create container");
	    } else if (rc) {
		    DCHECK(rc, "Failed to create container");
	    }

	    rc = dfs_mount(hdl.poh, hdl.coh, O_RDWR, &hdl.dfs);
	    DCHECK(rc, "Failed DFS mount");
    }

    rc = share_handles(rank, MPI_COMM_WORLD);
    DCHECK(rc, "Failed to share handles");

    /* Run the tests */
    for (k = 0; k < set_iterations; k++) {
	if ((rank == 0) && (set_iterations > 1))
	    printf("%s: Set iteration %d\n", timestamp(), k);
	for (i = 0; ; ++i) {
	    test = incl_list[i];
	    if (test == -1)
		break;
	    if (!int_in_list(test, excl_list)) {
		for (j = 0; j < iterations; j++) {
		    if (singlestep) {
			if (rank == 0)
			    printf("%s: Hit <Enter> to run test #%d(iter %d): %s, %s mode.\n",
				   timestamp(), test, j,
				   testlist[test%testcnt].name,
				   (test < testcnt) ? "shared" : "individual");
			fgets(linebuf, 80, stdin);
		    }
		    if (rank == 0) {
			printf("%s: Running test #%d(iter %d): %s, %s mode.\n",
			       timestamp(), test, j, testlist[test%testcnt].name,
			       (test < testcnt) ? "shared" : "individual");
			fflush(stdout);
		    }
		    testlist[test%testcnt].function((test < testcnt) ? SHARED : !SHARED);
		    MPI_Barrier(MPI_COMM_WORLD);
		}
	    }
	}
    }

    shutdown_daos();

    if (rank == 0) printf("%s: All tests passed!\n", timestamp());
    MPI_Errhandler_free(&newerr);
    MPI_Finalize();
    exit(0);
}
