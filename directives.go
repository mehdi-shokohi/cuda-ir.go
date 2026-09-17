package cudair

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strconv"
	"strings"
)

// directive holds the `//cuda:` doc-comment directives of one kernel:
//
//	//cuda:launch_bounds 256      __launch_bounds__(256)
//	//cuda:launch_bounds 256 2    __launch_bounds__(256, 2)
//	//cuda:maxnreg 64             __maxnreg__(64)
//	//cuda:cluster_dims 2 [1 [1]] __cluster_dims__(2, 1, 1) (sm_90+)
//	//cuda:restrict               __restrict__ on every pointer parameter
//	//cuda:grid_constant          __grid_constant__ on every struct parameter
type directive struct {
	maxThreads, minBlocks, maxNReg int
	cluster                        [3]int
	restrict, gridConstant         bool
}

// attrs renders the directive as NVPTX function attributes.
func (d directive) attrs() string {
	var a []string
	if d.maxThreads > 0 {
		a = append(a, fmt.Sprintf(`"nvvm.maxntid"="%d"`, d.maxThreads))
	}
	if d.minBlocks > 0 {
		a = append(a, fmt.Sprintf(`"nvvm.minctasm"="%d"`, d.minBlocks))
	}
	if d.maxNReg > 0 {
		a = append(a, fmt.Sprintf(`"nvvm.maxnreg"="%d"`, d.maxNReg))
	}
	if d.cluster[0] > 0 {
		a = append(a, fmt.Sprintf(`"nvvm.cluster_dim"="%d,%d,%d"`, d.cluster[0], d.cluster[1], d.cluster[2]))
	}
	return strings.Join(a, " ")
}

// readDirectives parses the Go files of dir and collects the `//cuda:`
// directives from function doc comments, keyed by function name.
func readDirectives(dir string) (map[string]directive, error) {
	names, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil {
		return nil, err
	}
	fset := token.NewFileSet()
	out := map[string]directive{}
	for _, name := range names {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, name, nil, parser.ParseComments)
		if err != nil {
			return nil, err
		}
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Doc == nil {
				continue
			}
			var d directive
			has := false
			for _, c := range fn.Doc.List {
				text := strings.TrimPrefix(c.Text, "//cuda:")
				if text == c.Text {
					continue
				}
				fields := strings.Fields(text)
				if len(fields) == 0 {
					continue
				}
				nums := make([]int, 0, 2)
				for _, s := range fields[1:] {
					n, err := strconv.Atoi(s)
					if err != nil {
						return nil, fmt.Errorf("%s: %s: %v", fset.Position(c.Pos()), c.Text, err)
					}
					nums = append(nums, n)
				}
				switch fields[0] {
				case "launch_bounds":
					if len(nums) < 1 || len(nums) > 2 {
						return nil, fmt.Errorf("%s: //cuda:launch_bounds wants maxThreads [minBlocks]", fset.Position(c.Pos()))
					}
					d.maxThreads = nums[0]
					if len(nums) == 2 {
						d.minBlocks = nums[1]
					}
				case "maxnreg":
					if len(nums) != 1 {
						return nil, fmt.Errorf("%s: //cuda:maxnreg wants one number", fset.Position(c.Pos()))
					}
					d.maxNReg = nums[0]
				case "cluster_dims":
					if len(nums) < 1 || len(nums) > 3 {
						return nil, fmt.Errorf("%s: //cuda:cluster_dims wants x [y [z]]", fset.Position(c.Pos()))
					}
					d.cluster = [3]int{1, 1, 1}
					copy(d.cluster[:], nums)
				case "restrict":
					if len(nums) != 0 {
						return nil, fmt.Errorf("%s: //cuda:restrict takes no arguments", fset.Position(c.Pos()))
					}
					d.restrict = true
				case "grid_constant":
					if len(nums) != 0 {
						return nil, fmt.Errorf("%s: //cuda:grid_constant takes no arguments", fset.Position(c.Pos()))
					}
					d.gridConstant = true
				default:
					return nil, fmt.Errorf("%s: unknown directive //cuda:%s", fset.Position(c.Pos()), fields[0])
				}
				has = true
			}
			if has {
				out[fn.Name.Name] = d
			}
		}
	}
	return out, nil
}
