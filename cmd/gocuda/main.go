// gocuda build compiles a Go package of kernels to PTX; the same pipeline
// is available from Go as package cudair (github.com/mehdi-shokohi/cuda-ir.go).
//
//	gocuda build [-o out.ptx] [-sm sm_80] [-ptx 78] [-kernel A,B] [-O 2] [-keep] [-v] ./pkg
//	gocuda doctor            # check that every tool gocuda needs is installed
package main

import (
	"flag"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/mehdi-shokohi/cuda-ir.go"
)

func main() {
	if len(os.Args) >= 2 && os.Args[1] == "doctor" {
		if err := cudair.Doctor(os.Stdout); err != nil {
			os.Exit(1)
		}
		return
	}
	if len(os.Args) < 2 || os.Args[1] != "build" {
		fmt.Fprintln(os.Stderr, "usage: gocuda build [-o out.ptx] [-sm sm_80] [-ptx 78] [-kernel A,B] [-O 2] [-keep] [-v] ./pkg\n       gocuda doctor")
		os.Exit(2)
	}
	fs := flag.NewFlagSet("build", flag.ExitOnError)
	outFile := fs.String("o", "", "output .ptx (default <pkg>.ptx in the current directory)")
	sm := fs.String("sm", "sm_80", "target compute capability for llc/ptxas")
	ptxVer := fs.String("ptx", "78", "PTX ISA version to emit (73 minimum)")
	kernelList := fs.String("kernel", "", "comma-separated kernel names (default: all exported void funcs of the package)")
	optLevel := fs.String("O", "2", "opt level passed to opt (0 to skip)")
	keep := fs.Bool("keep", false, "keep intermediate .ll files next to the output")
	check := fs.Bool("check", true, "run ptxas on the result if available")
	verbose := fs.Bool("v", false, "print commands")
	fs.Parse(os.Args[2:])
	if fs.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "gocuda: exactly one package argument is required")
		os.Exit(1)
	}
	if os.Getenv("LLGO_ROOT") == "" {
		fmt.Fprintln(os.Stderr, "gocuda: warning: LLGO_ROOT is not set; llgen may fail")
	}
	opts := &cudair.Options{SM: *sm, PTX: *ptxVer, Opt: *optLevel, NoCheck: !*check}
	if *kernelList != "" {
		opts.Kernels = strings.Split(*kernelList, ",")
	}
	if *keep {
		opts.WorkDir = "."
		if *outFile != "" {
			opts.WorkDir = filepath.Dir(*outFile)
		}
	}
	if *verbose {
		opts.Log = os.Stderr
	}
	res, err := cudair.BuildFile(fs.Arg(0), *outFile, opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gocuda: %v\n", err)
		os.Exit(1)
	}
	out := *outFile
	if out == "" {
		out = path.Base(res.Package) + ".ptx"
	}
	fmt.Printf("%s: kernels %s\n", out, strings.Join(res.Kernels, ", "))
}
