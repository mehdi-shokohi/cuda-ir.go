package cuda

import _ "unsafe" // for go:linkname

// ---- special registers beyond the thread/block indices

// SMID is %smid: the id of the streaming multiprocessor running the thread.
//
//go:linkname SMID llvm.nvvm.read.ptx.sreg.smid
func SMID() int32

// NumSMs is %nsmid: the number of SMs (upper bound of SMID).
//
//go:linkname NumSMs llvm.nvvm.read.ptx.sreg.nsmid
func NumSMs() int32

// WarpID is %warpid: the warp's slot in its SM (may change on context
// switches; use ThreadIdxX()/32 for a stable warp index within the block).
//
//go:linkname WarpID llvm.nvvm.read.ptx.sreg.warpid
func WarpID() int32

// NumWarps is %nwarpid.
//
//go:linkname NumWarps llvm.nvvm.read.ptx.sreg.nwarpid
func NumWarps() int32

// GridID is %gridid, a per-launch id (the low 32 bits).
//
//go:linkname GridID llvm.nvvm.read.ptx.sreg.gridid
func GridID() int32

// LaneMaskEq/Lt/Le/Gt/Ge are %lanemask_eq/lt/le/gt/ge: bit masks of the
// lanes equal to / before / after this one.
//
//go:linkname LaneMaskEq llvm.nvvm.read.ptx.sreg.lanemask.eq
func LaneMaskEq() uint32

//go:linkname LaneMaskLt llvm.nvvm.read.ptx.sreg.lanemask.lt
func LaneMaskLt() uint32

//go:linkname LaneMaskLe llvm.nvvm.read.ptx.sreg.lanemask.le
func LaneMaskLe() uint32

//go:linkname LaneMaskGt llvm.nvvm.read.ptx.sreg.lanemask.gt
func LaneMaskGt() uint32

//go:linkname LaneMaskGe llvm.nvvm.read.ptx.sreg.lanemask.ge
func LaneMaskGe() uint32
