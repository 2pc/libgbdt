RC_DIR = .
IN_DIR = bin
UILD_DIR = bin/build

#ULIB Path
ORKROOT = ../..

LIBHOME=$(WORKROOT)/lib2-64
PUBLIC=$(WORKROOT)/lib2-64
VERSION = $(VERSION32_SIGN)

INCLUDE = -I$(SRC_DIR)/ \
 -I$(LIBHOME)/ullib/include

LIB =-L$(LIBHOME)/ullib/lib \
	 -lpthread

#LIB =-L$(LIBHOME)/ullib/lib \
# -lullib -lpthread

CC = g++

CPPFLAGS = -g -Wall -D_REENTRANT -O3 -fPIC
DEBUG_CPPFLAGS = -g -Wall -fPIC -D_REENTRANT -DDEBUG_ON -DFILE_DUMP_ON

all : libgbdt.a gbdt-train gbdt-test gbdt.so output

libgbdt.a : gradient_boosting.o
	ar rcv $@ $^

gbdt-train : gbdt_train.o libgbdt.a
	$(CC) -o $@ $^ $(INCLUDE) $(LIB) -DTEST_MAIN $(DEBUG_CPPFLAGS)

gbdt-test : gbdt_test.o libgbdt.a
	$(CC) -o $@ $^ $(INCLUDE) $(LIB) -DTEST_MAIN $(DEBUG_CPPFLAGS)

gbdt.so : gbdt_rabbit.o libgbdt.a
	$(CC) -shared -o $@ $^ $(LIB) -DTEST_MAIN $(DEBUG_CPPFLAGS)

%.o : %.cpp
	$(CC) -c -o $@ $^ $(INCLUDE) $(DEBUG_CPPFLAGS) $(INCLUDE)

OUTPUT_DIR = ./output
LIB_DIR = $(OUTPUT_DIR)/lib
INCLUDE_DIR = $(OUTPUT_DIR)/include
TEST_DIR = $(OUTPUT_DIR)/test

clean : 
	find . -name "*.o" -maxdepth 3 -exec rm {} \;
	find . -name "*.a" -maxdepth 3 -exec rm {} \;
	find . -name "*.so" -maxdepth 3 -exec rm {} \;
	find . -name "gbdt-train" -maxdepth 3 -exec rm {} \;
	find . -name "gbdt-test" -maxdepth 3 -exec rm {} \;
	if [ -d $(OUTPUT_DIR) ] ; then rm -r $(OUTPUT_DIR); fi

output : libgbdt.a gradient_boosting.h gbdt-train gbdt-test gbdt.so
	if [ ! -d $(OUTPUT_DIR) ] ; then mkdir $(OUTPUT_DIR) ; fi
	if [ ! -d $(LIB_DIR) ] ; then mkdir $(LIB_DIR) ; fi
	if [ ! -d $(INCLUDE_DIR) ] ; then mkdir $(INCLUDE_DIR) ; fi
	if [ ! -d $(TEST_DIR) ] ; then mkdir $(TEST_DIR) ; fi

	cp gbdt-train $(TEST_DIR)
	cp gbdt-test $(TEST_DIR)
	cp gbdt.so $(TEST_DIR)
	cp libgbdt.a $(LIB_DIR)
	cp gradient_boosting.h $(INCLUDE_DIR)
