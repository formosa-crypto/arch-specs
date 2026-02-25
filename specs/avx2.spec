# Intel intrinsic: _mm256_broadcastq_epi64
# Note: input should be @128
VPBROADCAST_4u64(w@64) -> @256 = 
  repeat<64>(w, 4)

# Intel intrinsic: _mm256_broadcastd_epi32
# Note: input should be @128
VPBROADCAST_8u32(w@32) -> @256 = 
  repeat<32>(w, 8)

# Intel intrinsic: _mm256_broadcastw_epi16
# Note: input should be @128
VPBROADCAST_16u16(w@16) -> @256 = 
  repeat<16>(w, 16)

# Intel intrinsic: _mm256_broadcastsi128_si256
VPBROADCAST_2u128(w@128) -> @256 = 
  repeat<128>(w, 2)

# Intel intrinsic: _mm256_permutevar8x32_epi32
# Note: arguments are swapped
VPERMD(widx@256, w@256) -> @256 =
  map<32, 8>(
    fun idx@32 . let i = idx[0:3] in w[@32|i],
    widx
  )

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

# Intel intrinsic: _mm256_srl_epi64
VPSRL_4u64(w@256, count@128) -> @256 =
  ugt<64>(count[@64|0], 63)
    ? 0
    : map<64, 4>(srl<64>(., count[@8|0]), w)

# Intel intrinsic: _mm256_srl_epi32
VPSRL_8u32(w@256, count@128) -> @256 =
  ugt<64>(count[@64|0], 31)
    ? 0
    : map<32, 8>(srl<32>(., count[@8|0]), w)

# Intel intrinsic: _mm256_srl_epi16
VPSRL_16u16(w@256, count@128) -> @256 =
  ugt<64>(count[@64|0], 15)
    ? 0
    : map<16, 16>(srl<16>(., count[@8|0]), w)

# Intel intrinsic: _mm256_sra_epi32
VPSRA_8u32(w@256, count@128) -> @256 =
  let c = ugt<64>(count[@64|0], 31) ? 31 : count[@8|0] in
  map<32, 8>(sra<32>(., c), w)

# Intel intrinsic: _mm256_sra_epi16
VPSRA_16u16(w@256, count@128) -> @256 =
  let c = ugt<64>(count[@64|0], 15) ? 15 : count[@8|0] in
  map<16, 16>(sra<16>(., c), w)

# Intel intrinsic: _mm256_sll_epi64
VPSLL_4u64(w@256, count@128) -> @256 =
  ugt<64>(count[@64|0], 63)
    ? 0
    : map<64, 4>(sll<64>(., count[@8|0]), w)

# Intel intrinsic: _mm256_sll_epi32
VPSLL_8u32(w@256, count@128) -> @256 =
  ugt<64>(count[@64|0], 31)
    ? 0
    : map<32, 8>(sll<32>(., count[@8|0]), w)

# Intel intrinsic: _mm256_sll_epi16
VPSLL_16u16(w@256, count@128) -> @256 =
  ugt<64>(count[@64|0], 15)
    ? 0
    : map<16, 16>(sll<16>(., count[@8|0]), w)

# Intel intrinsic: _mm256_sllv_epi64
VPSLLV_4u64(w@256, counts@256) -> @256 =
  map<64, 4>(
    fun w@64 count@64 .
      uge<64>(count, 64) ? 0 : sll<64>(w, count[@8|0]),
    w,
    counts
  )

# Intel intrinsic: _mm256_sllv_epi32
VPSLLV_8u32(ws@256, counts@256) -> @256 =
  map<32, 8>(
    fun w@32 count@32 .
      uge<32>(count, 32) ? 0 : sll<32>(w, count[@8|0]),
    ws,
    counts
  )

# Intel intrinsic: _mm256_srlv_epi64
VPSRLV_4u64(ws@256, counts@256) -> @256 =
  map<64, 4>(
    fun w@64 count@64 .
      uge<64>(count, 64) ? 0 : srl<64>(w, count[@8|0]),
    ws,
    counts
  )

# Intel intrinsic: _mm256_srlv_epi32
VPSRLV_8u32(ws@256, counts@256) -> @256 =
  map<32, 8>(
    fun w@32 count@32 .
      uge<32>(count, 32) ? 0 : srl<32>(w, count[@8|0]),
    ws,
    counts
  )

# Intel intrinsic: _mm256_unpacklo_epi8
VPUNPCKL_32u8(w1@256, w2@256) -> @256 =
  let interleave (w1@64, w2@64) =
    map<8, 8>(
      fun w1@8 w2@8 . concat<8>(w1, w2),
      w1,
      w2
    )
  in

  concat<128>(
    interleave(w1[@64|0], w2[@64|0]),
    interleave(w1[@64|2], w2[@64|2])
  )

# Intel intrinsic: _mm256_unpacklo_epi16
VPUNPCKL_16u16(w1@256, w2@256) -> @256 =
  let interleave (w1@64, w2@64) =
    map<16, 4>(
      fun w1@16 w2@16 . concat<16>(w1, w2),
      w1,
      w2
    )
  in

  concat<128>(
    interleave(w1[@64|0], w2[@64|0]),
    interleave(w1[@64|2], w2[@64|2])
  )

# Intel intrinsic: _mm256_unpacklo_epi16
VPUNPCKL_8u32(w1@256, w2@256) -> @256 =
  let interleave (w1@64, w2@64) =
    map<32, 2>(
      fun w1@32 w2@32 . concat<32>(w1, w2),
      w1,
      w2
    )
  in

  concat<128>(
    interleave(w1[@64|0], w2[@64|0]),
    interleave(w1[@64|2], w2[@64|2])
  )

# Intel intrinsic: _mm256_unpacklo_epi64
VPUNPCKL_4u64(w1@256, w2@256) -> @256 =
  let interleave (w1@128, w2@128) =
    concat<64>(w1[@64|0], w2[@64|0])
  in

  concat<128>(
    interleave(w1[@128|0], w2[@128|0]),
    interleave(w1[@128|1], w2[@128|1])
  )

# Intel intrinsic: _mm256_unpacklo_epi8
VPUNPCKH_32u8(w1@256, w2@256) -> @256 =
  let interleave (w1@64, w2@64) =
    map<8, 8>(
      fun w1@8 w2@8 . concat<8>(w1, w2),
      w1,
      w2
    )
  in

  concat<128>(
    interleave(w1[@64|1], w2[@64|1]),
    interleave(w1[@64|3], w2[@64|3])
  )

# Intel intrinsic: _mm256_unpacklo_epi16
VPUNPCKH_16u16(w1@256, w2@256) -> @256 =
  let interleave (w1@64, w2@64) =
    map<16, 4>(
      fun w1@16 w2@16 . concat<16>(w1, w2),
      w1,
      w2
    )
  in

  concat<128>(
    interleave(w1[@64|1], w2[@64|1]),
    interleave(w1[@64|3], w2[@64|3])
  )

# Intel intrinsic: _mm256_unpacklo_epi16
VPUNPCKH_8u32(w1@256, w2@256) -> @256 =
  let interleave (w1@64, w2@64) =
    map<32, 2>(
      fun w1@32 w2@32 . concat<32>(w1, w2),
      w1,
      w2
    )
  in

  concat<128>(
    interleave(w1[@64|1], w2[@64|1]),
    interleave(w1[@64|3], w2[@64|3])
  )

# Intel intrinsic: _mm256_unpackhi_epi64
VPUNPCKH_4u64(w1@256, w2@256) -> @256 =
  let interleave (w1@128, w2@128) =
    concat<64>(w1[@64|1], w2[@64|1])
  in

  concat<128>(
    interleave(w1[@128|0], w2[@128|0]),
    interleave(w1[@128|1], w2[@128|1])
  )
