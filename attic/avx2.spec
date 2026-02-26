
VEXTRACTI128(w@256, i@8) -> @128 =
  w[@128|i[0]]

VEXTRACTI32_256(w@256, i@8) -> @32 =
  w[@32|i[@3|0]]

VEXTRACTI32_128(w@128, i@8) -> @32 =
  w[@32|i[@2|0]]

VEXTRACTI32_64(w@64, i@8) -> @32 =
  w[@32|i[0]]


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

