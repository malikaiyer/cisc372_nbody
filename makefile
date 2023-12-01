FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

nbody: nbody.o compute.o
	nvcc $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.cu planets.h config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $< 
compute.o: compute.cu config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $< 
clean:
	rm -f *.o nbody

serial_nbody: serial_nbody.o serial_compute.o
	gcc $(FLAGS) $^ -o $@ $(LIBS)
serial_nbody.o: serial_nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $<
serial_compute.o: serial_compute.c config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $<
serial_clean:
	rm -f *.o serial_nbody 
