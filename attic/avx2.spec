# See: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
# 256 bit logical and (maybe change in ecBDep to VPAND)

# CHECKME
VPINSR_8u16(w1@128, w2@16, i@8) -> @128 =
  w1[@16|i[@3|0] <- w2]

# CHECKME
VPINSR_2u64(w1@128, w2@64, i@8) -> @128 =
  w1[@64|i[@1|0] <- w2]

#CHECKME
# Intel intrinsic: _mm256_permute2x128_si256
VPERM2I128(src1@256, src2@256, i@8) -> @256 =
  map<4,2>(
    fun control@4 .
        control[3] ? 0@128 :
          let w1 = concat<256>(src1,src2) in
          let w2 = w1[@128|control[@2|0]] in w2,
      i)

VPSLLDQ_256(w@256, count@8) -> @256 =
  map<128, 2>(sll<128>(., sll<8>(count, 3)), w)

# CHECKME
VPSRLDQ_256(w@256, count@8) -> @256 =
  map<128, 2>(srl<128>(., sll<8>(count, 3)), w)

VPSLLDQ_128(w@128, count@8) -> @128 =
  sll<128>(w, sll<8>(count, 3))

VPSRLDQ_128(w@128, count@8) -> @128 =
  srl<128>(w, sll<8>(count, 3))

# Intel intrinsic: _mm256_maddubs_epi16
VPMADDUBSW_256(w1@256, w2@256) -> @256 =
  map<16, 16>(
    fun x@16 y@16 .
      ssadd<16>(
        usmul<8>(x[@8|0], y[@8|0]),
        usmul<8>(x[@8|1], y[@8|1])
      ),
    w1,
    w2
  )

VPMADDWD_16u16(w1@256, w2@256) -> @256 =
  map<32, 8>(
    fun x@32 y@32 .
      add<32>(
        smul<16>(x[@16|0], y[@16|0]),
        smul<16>(x[@16|1], y[@16|1])
      ),
    w1,
    w2
  )

# Intel intrinsic: _mm256_packus_epi16
VPACKUS_16u16(w1@256, w2@256) -> @256 =
  let pack (w@128) = map<16, 8>(usat<16, 8>, w) in

  concat<64>(
    pack(w1[@128|0]),
    pack(w2[@128|0]),
    pack(w1[@128|1]),
    pack(w2[@128|1])
  )

# Intel intrincis: _mm256_packs_epi16
VPACKSS_16u16(w1@256, w2@256) -> @256 =
  let pack (w@128) = map<16, 8>(ssat<16, 8>, w) in

  concat<64>(
    pack(w1[@128|0]),
    pack(w2[@128|0]),
    pack(w1[@128|1]),
    pack(w2[@128|1])
  )



# https://www.felixcloutier.com/x86/pshufd
VPSHUFD_128(w@128, idx@8) -> @128 =
  concat<32>(
    w[@32|idx[@2|0]],
    w[@32|idx[@2|1]],
    w[@32|idx[@2|2]],
    w[@32|idx[@2|3]]
  )

# Intel intrinsic: _mm256_blend_epi32
# FIXME: we need an heterogeneous `map' combinator
VPBLEND_8u32(w1@256, w2@256, c@8) -> @256 =
  let c = map<1, 8>(uextend<1, 32>, c) in

  map<32, 8>(
    fun c@32 w1@32 w2@32 . c[0] ? w2 : w1,
    c,
    w1,
    w2
  )

# CHECKME
# Intel intrinsic: _mm256_blend_epi32
# FIXME: we need an heterogeneous `map' combinator
VPBLEND_32u8(w1@256, w2@256, c@256) -> @256 =
  map<8, 32>(
    fun c@8 w1@8 w2@8 . c[7] ? w2 : w1,
    c,
    w1,
    w2
  )

# CHECKME
# Intel intrinsic: _mm256_blend_epi32
# FIXME: we need an heterogeneous `map' combinator
VPBLEND_16u8(w1@128, w2@128, c@128) -> @128 =
  map<8, 16>(
    fun c@8 w1@8 w2@8 . c[7] ? w2 : w1,
    c,
    w1,
    w2
  )

# Intel intrinsic: _mm256_blend_epi16
# FIXME: we need an heterogeneous `map' combinator
VPBLEND_16u16(w1@256, w2@256, c@8) -> @256 =
  let c = repeat<8>(c, 2) in
  let c = map<1, 16>(uextend<1, 16>, c) in

  map<16, 16>(
    fun c@16 w1@16 w2@16 . c[0] ? w2 : w1,
    c,
    w1,
    w2
  )

# CHECKME
VPBLENDW_128(w1@128, w2@128, c@8) -> @128 =
  let c = map<1, 8>(uextend<1, 16>, c) in

  map<16, 8>(
    fun c@16 w1@16 w2@16 . c[0] ? w2 : w1,
    c,
    w1,
    w2
  )

# CHECKME
VPBLENDW_256(w1@256, w2@256, c@8) -> @256 =
  let vpblend128(w1@128, w2@128)  =
    let c = map<1, 8>(uextend<1, 16>, c) in
    map<16, 8>(
     fun c@16 w1@16 w2@16 . c[0] ? w2 : w1,
      c,
      w1,
      w2
  ) in map<128, 2>(vpblend128,w1,w2)

VPBLENDW_128(w1@128, w2@128, c@8) -> @128 =
  let c = map<1, 8>(uextend<1, 16>, c) in
  map<16, 8>(
   fun c@16 w1@16 w2@16 . c[0] ? w2 : w1,
    c,
    w1,
    w2
  ) 

VPBLENDVB_128(w1@128, w2@128, c@128) -> @128 =
  map<8, 16>(
    fun c@8 w1@8 w2@8 . c[7] ? w2 : w1,
    c,
    w1,
    w2
  )

# Intel intrincis: _mm256_movemask_epi8
VPMOVMSKB_u256u64(w@256) -> @64 =
  uextend<32, 64>(map<8, 32>(fun i@8 . i[7], w))

VPMOVMSKB_u256u32(w@256) -> @32 =
  map<8, 32>(fun i@8 . i[7], w)



#CHECKME
VMOVSLDUP_256(a@256) -> @256 =
  let vmovsldup128(v@128)=
      concat<32>(v[@32|0],v[@32|0],v[@32|2],v[@32|2])
  in map<128,2>(vmovsldup128,a)

# Intel intrinsic: _mm256_extracti128_si256
VPEXTRACTI128(w@256, i@8) -> @128 =
  w[@128|i[0]]

VEXTRACTI128(w@256, i@8) -> @128 =
  w[@128|i[0]]

VEXTRACTI32_256(w@256, i@8) -> @32 =
  w[@32|i[@3|0]]

VEXTRACTI32_128(w@128, i@8) -> @32 =
  w[@32|i[@2|0]]

VEXTRACTI32_64(w@64, i@8) -> @32 =
  w[@32|i[0]]

# Intel intrinsic: _mm256_inserti128_si256
VPINSERTI128(w@256, m@128, i@8) -> @256 =
  w[@128|i[0] <- m]

# FIXME
concat_2u128(a@128, b@128) -> @256 =
  concat<128>(b, a)

# Add later
truncateu128(w@256) -> @128 =
  w[@128|0]

## Auxiliary stuff
COMPRESS(w@16) -> @4 =
  srl<32>(umul<16>(
    add<16>(
      sll<16>(w, 4), 
      1665)
  , 80635), 28)[@4|0]

## EASYCRYPT WORD OPERATORS

## INT CONVERSIONS
## Assuming INT = 256 bits
TO_UINT8(a@8) -> @256 =
  uextend<8, 256>(a)

TO_UINT16(a@16) -> @256 =
  uextend<16, 256>(a)
  
TO_UINT32(a@32) -> @256 =
  uextend<32, 256>(a)

TO_UINT64(a@64) -> @256 =
  uextend<64, 256>(a)

TO_UINT128(a@128) -> @256 =
  uextend<128, 256>(a)

TO_UINT256(a@256) -> @256 =
  uextend<256, 256>(a)

OF_INT8(a@256) -> @8 =
  a[@8|0]

OF_INT16(a@256) -> @16 =
  a[@16|0]

OF_INT32(a@256) -> @32 =
  a[@32|0]

OF_INT64(a@256) -> @64 =
  a[@64|0]

OF_INT128(a@256) -> @128 =
  a[@128|0]

OF_INT256(a@256) -> @256 =
  a[@256|0]

LSHIFT32(a@32, c@8) -> @32 =
  sll<32>(a, c)

RSHIFTL_8(a@8, c@8) -> @8 =
  srl<8>(a, c)

RSHIFTA_8(a@8, c@8) -> @8 =
  sra<8>(a, c)

RSHIFTL_16(a@16, c@8) -> @16 =
  srl<16>(a, c)

RSHIFTA_16(a@16, c@8) -> @16 =
  sra<16>(a, c)

RSHIFTL_32(a@32, c@8) -> @32 =
  srl<32>(a, c)

RSHIFTA_32(a@32, c@8) -> @32 =
  sra<32>(a, c)

LT_256(a@256, b@256) -> @1 =
  ugt<256>(b, a)

POPCOUNT_64(x@64) -> @64 =
  popcount<64, 64>(x)

# Not part of the arch
VPINC_8u8(w@64) -> @64 =
  map<8, 8>(incr<8>, w)


VPSHUFB_128(w@128, widx@128) -> @128 =
  map<8, 16>(
    fun idx@8 . idx[7] ? 0 : w[@8|idx[@4|0]],
    widx
  )

# MAPREDUCE EXAMPLE
XOR_LEFT8(w@8) -> @8 =
  xor<8>(xor<8>(w, 42@8), 213@8)

XOR_RIGHT8(w@8) -> @8 =
  xor<8>(w, xor<8>(42@8, 213@8))

