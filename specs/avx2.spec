# Intel intrinsic: _mm2_and_si128
VPAND_128(w1@128, w2@128) -> @128 = 
  and<128>(w1, w2)

# Intel intrinsic: _mm256_and_si256
VPAND_256(w1@256, w2@256) -> @256 = 
  and<256>(w1, w2)

# Intel intrinsic: _mm_andnot_si128
VPANDN_128(w1@128, w2@128) -> @128 = 
  and<128>(not<128>(w1), w2)

# Intel intrinsic: _mm256_andnot_si256
VPANDN_256(w1@256, w2@256) -> @256 = 
  and<256>(not<256>(w1), w2)

# Intel intrinsic: _mm_or_si128
VPOR_128(w1@128, w2@128) -> @128 =
  or<128>(w1, w2)

# Intel intrinsic: _mm256_or_si256
VPOR_256(w1@256, w2@256) -> @256 =
  or<256>(w1, w2)

# Intel intrinsic: _mm256_xor_si256
VPXOR_256(w1@256, w2@256) -> @256 =
  xor<256>(w1, w2)

# Intel intrinsic: _mm_xor_si128
VPXOR_128(w1@128, w2@128) -> @128 =
  xor<128>(w1, w2)

# Intel intrinsic: _mm256_add_epi64
VPADD_4u64(w1@256, w2@256) -> @256 =
  map<64, 4>(add<64>, w1, w2)

# Intel intrinsic: _mm256_add_epi32
VPADD_8u32(w1@256, w2@256) -> @256 =
  map<32, 8>(add<32>, w1, w2)

# Intel intrinsic: _mm256_add_epi16
VPADD_16u16(w1@256, w2@256) -> @256 =
  map<16, 16>(add<16>, w1, w2)

# Intel intrinsic: _mm256_add_epi8
VPADD_32u8(w1@256, w2@256) -> @256 =
  map<8, 32>(add<8>, w1, w2)

# Intel intrinsic: _mm256_sub_epi64
VPSUB_4u64(w1@256, w2@256) -> @256 =
  map<64, 4>(sub<64>, w1, w2)

# Intel intrinsic: _mm256_sub_epi32
VPSUB_8u32(w1@256, w2@256) -> @256 =
  map<32, 8>(sub<32>, w1, w2)

# Intel intrinsic: _mm256_sub_epi16
VPSUB_16u16(w1@256, w2@256) -> @256 =
  map<16, 16>(sub<16>, w1, w2)

# Intel intrinsic: _mm256_sub_epi8
VPSUB_32u8(w1@256, w2@256) -> @256 =
  map<8, 32>(sub<8>, w1, w2)

# Intel intrinsic: _mm256_mullo_epi16
VPMULL_16u16(w1@256, w2@256) -> @256 =
  map<16, 16>(smullo<16>, w1, w2)

# Intel intrinsic: _mm256_mulhi_epi16
VPMULH_16u16(w1@256, w2@256) -> @256 =
  map<16, 16>(smulhi<16>, w1, w2)

# Intel intrinsic: _mm256_mulhi_epu16
VPMULHU_16u16(w1@256, w2@256) -> @256 =
  map<16, 16>(umulhi<16>, w1, w2)

# Intel intrinsic: _mm256_mulhrs_epi16
VPMULHRS_16u16(w1@256, w2@256) -> @256 =
  map<16, 16>(
    fun x@16 y@16 .
      let w = smul<16>(x, y) in
      let w = incr<32>(srl<32>(w, 14)) in
      w[1:16],
    w1,
    w2
  )
