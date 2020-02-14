DAOS_DIR := /home/mschaara/install/daos
CART_DIR := /home/mschaara/install/deps_daos/cart

CC := mpicc
LDFLAGS := -L${CART_DIR}/lib -L${DAOS_DIR}/lib64 -ldaos -ldfs -ldaos_common -lgurt -luuid
INCLUDES := -I${DAOS_DIR}/include -I${CART_DIR}/include

build:
	mpicc -Wall -o simul simul.c
	${CC} -Wall -g ${INCLUDES} simul_dfs.c -o simul_dfs ${LDFLAGS}

clean:
	rm -f simul simul.o simul_dfs.o simul_dfs
