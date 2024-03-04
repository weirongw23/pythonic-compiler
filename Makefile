all: runtime/stub.exe

runtime/stub.exe: runtime/stub.rs runtime/libcompiled_code.a
	rustc -L runtime -o runtime/stub.exe runtime/stub.rs

runtime/libcompiled_code.a: runtime/compiled_code.o
	ar rus runtime/libcompiled_code.a runtime/compiled_code.o

runtime/compiled_code.o: runtime/compiled_code.s
	nasm -felf64 -o runtime/compiled_code.o runtime/compiled_code.s


clean:
	rm -f runtime/*.o runtime/*.a runtime/*.exe
