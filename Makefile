LLGO_ROOT ?= $(HOME)/llgo
export LLGO_ROOT

.PHONY: deps install doctor examples test clean

deps:               ## install LLVM 22, llgo/llgen, gocuda (asks before sudo)
	./install.sh

install:            ## install the gocuda command
	go install ./cmd/gocuda

doctor: install     ## check all dependencies
	gocuda doctor

examples: install   ## compile the example kernels to PTX
	gocuda build -o examples/vecadd/vecadd.ptx ./examples/vecadd
	gocuda build -o examples/features/features.ptx ./examples/features

test: examples      ## run the examples on the GPU (no cgo needed)
	CGO_ENABLED=0 go run ./examples/vecadd/run examples/vecadd/vecadd.ptx
	CGO_ENABLED=0 go run ./examples/features/run examples/features/features.ptx

clean:
	rm -f examples/*/*.ptx *.ptx *.ll
