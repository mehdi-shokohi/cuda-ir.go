package cuda

import _ "unsafe" // for go:linkname

// ---- cache-hint loads and stores (__ldca/__ldcg/__ldcs/__ldlu/__ldcv,
// __stwb/__stcg/__stcs/__stwt): the pointer must be global memory.
//
//	ca  cache at all levels (default)    wb  write back (default)
//	cg  cache in L2 only, bypass L1      cg  cache in L2 only
//	cs  streaming, evict first           cs  streaming
//	lu  last use                         wt  write through to system memory
//	cv  do not cache, fetch again

//go:linkname loadCAI32 cudair.ld.ca.i32
func loadCAI32(p *int32) int32

//go:linkname loadCAI64 cudair.ld.ca.i64
func loadCAI64(p *int64) int64

//go:linkname loadCAF32 cudair.ld.ca.float
func loadCAF32(p *float32) float32

//go:linkname loadCAF64 cudair.ld.ca.double
func loadCAF64(p *float64) float64

//go:linkname loadCGI32 cudair.ld.cg.i32
func loadCGI32(p *int32) int32

//go:linkname loadCGI64 cudair.ld.cg.i64
func loadCGI64(p *int64) int64

//go:linkname loadCGF32 cudair.ld.cg.float
func loadCGF32(p *float32) float32

//go:linkname loadCGF64 cudair.ld.cg.double
func loadCGF64(p *float64) float64

//go:linkname loadCSI32 cudair.ld.cs.i32
func loadCSI32(p *int32) int32

//go:linkname loadCSI64 cudair.ld.cs.i64
func loadCSI64(p *int64) int64

//go:linkname loadCSF32 cudair.ld.cs.float
func loadCSF32(p *float32) float32

//go:linkname loadCSF64 cudair.ld.cs.double
func loadCSF64(p *float64) float64

//go:linkname loadLUI32 cudair.ld.lu.i32
func loadLUI32(p *int32) int32

//go:linkname loadLUI64 cudair.ld.lu.i64
func loadLUI64(p *int64) int64

//go:linkname loadLUF32 cudair.ld.lu.float
func loadLUF32(p *float32) float32

//go:linkname loadLUF64 cudair.ld.lu.double
func loadLUF64(p *float64) float64

//go:linkname loadCVI32 cudair.ld.cv.i32
func loadCVI32(p *int32) int32

//go:linkname loadCVI64 cudair.ld.cv.i64
func loadCVI64(p *int64) int64

//go:linkname loadCVF32 cudair.ld.cv.float
func loadCVF32(p *float32) float32

//go:linkname loadCVF64 cudair.ld.cv.double
func loadCVF64(p *float64) float64

//go:linkname storeWBI32 cudair.st.wb.i32
func storeWBI32(p *int32, v int32)

//go:linkname storeWBI64 cudair.st.wb.i64
func storeWBI64(p *int64, v int64)

//go:linkname storeWBF32 cudair.st.wb.float
func storeWBF32(p *float32, v float32)

//go:linkname storeWBF64 cudair.st.wb.double
func storeWBF64(p *float64, v float64)

//go:linkname storeCGI32 cudair.st.cg.i32
func storeCGI32(p *int32, v int32)

//go:linkname storeCGI64 cudair.st.cg.i64
func storeCGI64(p *int64, v int64)

//go:linkname storeCGF32 cudair.st.cg.float
func storeCGF32(p *float32, v float32)

//go:linkname storeCGF64 cudair.st.cg.double
func storeCGF64(p *float64, v float64)

//go:linkname storeCSI32 cudair.st.cs.i32
func storeCSI32(p *int32, v int32)

//go:linkname storeCSI64 cudair.st.cs.i64
func storeCSI64(p *int64, v int64)

//go:linkname storeCSF32 cudair.st.cs.float
func storeCSF32(p *float32, v float32)

//go:linkname storeCSF64 cudair.st.cs.double
func storeCSF64(p *float64, v float64)

//go:linkname storeWTI32 cudair.st.wt.i32
func storeWTI32(p *int32, v int32)

//go:linkname storeWTI64 cudair.st.wt.i64
func storeWTI64(p *int64, v int64)

//go:linkname storeWTF32 cudair.st.wt.float
func storeWTF32(p *float32, v float32)

//go:linkname storeWTF64 cudair.st.wt.double
func storeWTF64(p *float64, v float64)

// LoadCAInt32 & co are __ldca (ld.global.ca).
func LoadCAInt32(p *int32) int32       { return loadCAI32(p) }
func LoadCAInt64(p *int64) int64       { return loadCAI64(p) }
func LoadCAFloat32(p *float32) float32 { return loadCAF32(p) }
func LoadCAFloat64(p *float64) float64 { return loadCAF64(p) }

// LoadCGInt32 & co are __ldcg (ld.global.cg: L2 only).
func LoadCGInt32(p *int32) int32       { return loadCGI32(p) }
func LoadCGInt64(p *int64) int64       { return loadCGI64(p) }
func LoadCGFloat32(p *float32) float32 { return loadCGF32(p) }
func LoadCGFloat64(p *float64) float64 { return loadCGF64(p) }

// LoadCSInt32 & co are __ldcs (ld.global.cs: streaming).
func LoadCSInt32(p *int32) int32       { return loadCSI32(p) }
func LoadCSInt64(p *int64) int64       { return loadCSI64(p) }
func LoadCSFloat32(p *float32) float32 { return loadCSF32(p) }
func LoadCSFloat64(p *float64) float64 { return loadCSF64(p) }

// LoadLUInt32 & co are __ldlu (ld.global.lu: last use).
func LoadLUInt32(p *int32) int32       { return loadLUI32(p) }
func LoadLUInt64(p *int64) int64       { return loadLUI64(p) }
func LoadLUFloat32(p *float32) float32 { return loadLUF32(p) }
func LoadLUFloat64(p *float64) float64 { return loadLUF64(p) }

// LoadCVInt32 & co are __ldcv (ld.global.cv: volatile, uncached).
func LoadCVInt32(p *int32) int32       { return loadCVI32(p) }
func LoadCVInt64(p *int64) int64       { return loadCVI64(p) }
func LoadCVFloat32(p *float32) float32 { return loadCVF32(p) }
func LoadCVFloat64(p *float64) float64 { return loadCVF64(p) }

// StoreWBInt32 & co are __stwb (st.global.wb).
func StoreWBInt32(p *int32, v int32)       { storeWBI32(p, v) }
func StoreWBInt64(p *int64, v int64)       { storeWBI64(p, v) }
func StoreWBFloat32(p *float32, v float32) { storeWBF32(p, v) }
func StoreWBFloat64(p *float64, v float64) { storeWBF64(p, v) }

// StoreCGInt32 & co are __stcg (st.global.cg).
func StoreCGInt32(p *int32, v int32)       { storeCGI32(p, v) }
func StoreCGInt64(p *int64, v int64)       { storeCGI64(p, v) }
func StoreCGFloat32(p *float32, v float32) { storeCGF32(p, v) }
func StoreCGFloat64(p *float64, v float64) { storeCGF64(p, v) }

// StoreCSInt32 & co are __stcs (st.global.cs).
func StoreCSInt32(p *int32, v int32)       { storeCSI32(p, v) }
func StoreCSInt64(p *int64, v int64)       { storeCSI64(p, v) }
func StoreCSFloat32(p *float32, v float32) { storeCSF32(p, v) }
func StoreCSFloat64(p *float64, v float64) { storeCSF64(p, v) }

// StoreWTInt32 & co are __stwt (st.global.wt).
func StoreWTInt32(p *int32, v int32)       { storeWTI32(p, v) }
func StoreWTInt64(p *int64, v int64)       { storeWTI64(p, v) }
func StoreWTFloat32(p *float32, v float32) { storeWTF32(p, v) }
func StoreWTFloat64(p *float64, v float64) { storeWTF64(p, v) }
