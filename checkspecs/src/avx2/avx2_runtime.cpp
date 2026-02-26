/* ==================================================================== */
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>

/* -------------------------------------------------------------------- */
#define NX(A, B) A ## B

/* -------------------------------------------------------------------- */
#ifdef SIMDe
# include <simde/x86/avx2.h>
# define SIMD(B) NX(simde_, B)
#else
# include <immintrin.h>
# define SIMD(B) NX(_, B)
#endif

/* -------------------------------------------------------------------- */
typedef SIMD(_m256i) m256i;
typedef SIMD(_m128i) m128i;

/* -------------------------------------------------------------------- */
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/fail.h>

/* ==================================================================== */
extern "C" CAMLprim value m64_of_32x2(value lohi) {
  CAMLparam1(lohi);

  const uint32_t lo = (uint32_t) Int32_val(Field(lohi, 0));
  const uint32_t hi = (uint32_t) Int32_val(Field(lohi, 1));

  const uint64_t out = ((uint64_t) lo) | (((uint64_t) hi) << 32);

  CAMLreturn(caml_copy_int64((int64_t) out));
}

/* -------------------------------------------------------------------- */
extern "C" CAMLprim value m64_to_32x2(value lohi) {
  CAMLparam1(lohi);
  CAMLlocal1(out);

  const uint64_t v = (uint64_t) Int64_val(lohi);

  const uint32_t lo = (v >>  0) & 0xffffffff;
  const uint32_t hi = (v >> 32) & 0xffffffff;

  out = caml_alloc_tuple(2);
  Field(out, 0) = caml_copy_int32(lo);
  Field(out, 1) = caml_copy_int32(hi);

  CAMLreturn(out);
}

/* -------------------------------------------------------------------- */
extern "C" CAMLprim value m32_of_16x2(value lohi) {
  CAMLparam1(lohi);

  const uint16_t lo = (uint16_t) Int_val(Field(lohi, 0));
  const uint16_t hi = (uint16_t) Int_val(Field(lohi, 1));

  const uint32_t out = ((uint32_t) lo) | (((uint32_t) hi) << 16);

  CAMLreturn(caml_copy_int32((int32_t) out));
}

/* -------------------------------------------------------------------- */
extern "C" CAMLprim value m32_to_16x2(value lohi) {
  CAMLparam1(lohi);
  CAMLlocal1(out);

  const uint32_t v = (uint32_t) Int32_val(lohi);

  const uint16_t lo = (v >>  0) & 0xffff;
  const uint16_t hi = (v >> 16) & 0xffff;

  out = caml_alloc_tuple(2);
  Field(out, 0) = Val_int(lo);
  Field(out, 1) = Val_int(hi);

  CAMLreturn(out);
}

/* -------------------------------------------------------------------- */
extern "C" CAMLprim value m16_of_8x2(value lohi) {
  CAMLparam1(lohi);

  const uint8_t lo = (uint8_t) Int_val(Field(lohi, 0));
  const uint8_t hi = (uint8_t) Int_val(Field(lohi, 1));

  const uint16_t out = ((uint16_t) lo) | (((uint16_t) hi) << 8);

  CAMLreturn(Val_int(out));
}

/* -------------------------------------------------------------------- */
extern "C" CAMLprim value m16_to_8x2(value lohi) {
  CAMLparam1(lohi);
  CAMLlocal1(out);

  const uint16_t v = (uint16_t) Int_val(lohi);

  const uint8_t lo = (v >> 0) & 0xff;
  const uint8_t hi = (v >> 8) & 0xff;

  out = caml_alloc_tuple(2);
  Field(out, 0) = Val_int(lo);
  Field(out, 1) = Val_int(hi);

  CAMLreturn(out);
}

/* -------------------------------------------------------------------- */
extern "C" CAMLprim value caml_avx2_using_simde(value unit) {
  CAMLparam1(unit);
#ifdef SIMDe
  CAMLreturn(Val_bool(1));
#else
  CAMLreturn(Val_bool(0));
#endif
}

/* ==================================================================== */
static value value_of_w256(m256i x) {
  CAMLparam0();
  CAMLlocal1(out);

  out = caml_alloc_tuple(4);
  Store_field(out, 0, caml_copy_int64(SIMD(mm256_extract_epi64)(x, 0)));
  Store_field(out, 1, caml_copy_int64(SIMD(mm256_extract_epi64)(x, 1)));
  Store_field(out, 2, caml_copy_int64(SIMD(mm256_extract_epi64)(x, 2)));
  Store_field(out, 3, caml_copy_int64(SIMD(mm256_extract_epi64)(x, 3)));

  CAMLreturn(out);
}

/* -------------------------------------------------------------------- */
static m256i w256_of_value(value x) {
  CAMLparam1(x);

  m256i out = SIMD(mm256_set_epi64x)(
    Int64_val(Field(x, 3)),
    Int64_val(Field(x, 2)),
    Int64_val(Field(x, 1)),
    Int64_val(Field(x, 0))
  );

  CAMLreturnT(m256i, out);
}

/* -------------------------------------------------------------------- */
static value value_of_w128(m128i x) {
  CAMLparam0();
  CAMLlocal1(out);

  out = caml_alloc_tuple(2);
  Store_field(out, 0, caml_copy_int64(SIMD(mm_extract_epi64)(x, 0)));
  Store_field(out, 1, caml_copy_int64(SIMD(mm_extract_epi64)(x, 1)));

  CAMLreturn(out);
}

/* -------------------------------------------------------------------- */
static m128i w128_of_value(value x) {
  CAMLparam1(x);

  m128i out = SIMD(mm_set_epi64x)(
    Int64_val(Field(x, 1)),
    Int64_val(Field(x, 0))
  );

  CAMLreturnT(m128i, out);
}

/* -------------------------------------------------------------------- */
struct M256i {
  typedef m256i type;

  static inline type ofocaml(value v) {
    return w256_of_value(v);
  }

  static inline value toocaml(type v) {
    return value_of_w256(v);
  }
};

/* -------------------------------------------------------------------- */
struct M128i {
  typedef m128i type;

  static inline type ofocaml(value v) {
    return w128_of_value(v);
  }

  static inline value toocaml(type v) {
    return value_of_w128(v);
  }
};

/* -------------------------------------------------------------------- */
struct Long {
  typedef long type;

  static inline type ofocaml(value v) {
    return Long_val(v);
  }

  static inline value toocaml(type v) {
    return Val_long(v);
  }
};

/* -------------------------------------------------------------------- */
struct Int32 {
  typedef long type;

  static inline type ofocaml(value v) {
    return Int32_val(v);
  }

  static inline value toocaml(type v) {
    return caml_copy_int32(v);
  }
};

/* -------------------------------------------------------------------- */
struct Int64 {
  typedef long type;

  static inline type ofocaml(value v) {
    return Int64_val(v);
  }

  static inline value toocaml(type v) {
    return caml_copy_int64(v);
  }
};

/* -------------------------------------------------------------------- */
template <auto F, typename U, typename T>
static value bind(value arg) {
  CAMLparam1(arg);
  typename T::type varg = T::ofocaml(arg);
  CAMLreturn(U::toocaml(F(varg)));
}

/* -------------------------------------------------------------------- */
template <auto F, typename U, typename T1, typename T2>
static value bind(value arg1, value arg2) {
  CAMLparam2(arg1, arg2);
  typename T1::type varg1 = T1::ofocaml(arg1);
  typename T2::type varg2 = T2::ofocaml(arg2);
  CAMLreturn(U::toocaml(F(varg1, varg2)));
}

/* -------------------------------------------------------------------- */
template <auto F, typename U, typename T1, typename T2, typename T3>
static value bind(value arg1, value arg2, value arg3) {
  CAMLparam3(arg1, arg2, arg3);
  typename T1::type varg1 = T1::ofocaml(arg1);
  typename T2::type varg2 = T2::ofocaml(arg2);
  typename T3::type varg3 = T3::ofocaml(arg3);
  CAMLreturn(U::toocaml(F(varg1, varg2, varg3)));
}

/* -------------------------------------------------------------------- */
template <auto F, typename U, typename T>
static value bindc(value arg, value imm8) {
  CAMLparam2(arg, imm8);
  typename T::type varg = T::ofocaml(arg);
  const uint8_t vimm8 = (uint8_t) Long_val(imm8);
  typename U::type aout;

#define CASE(I) case I: { aout = F(varg, I); break ; }

  switch (vimm8) {
  CASE(0x00); CASE(0x01); CASE(0x02); CASE(0x03); CASE(0x04); CASE(0x05); CASE(0x06); CASE(0x07)
  CASE(0x08); CASE(0x09); CASE(0x0a); CASE(0x0b); CASE(0x0c); CASE(0x0d); CASE(0x0e); CASE(0x0f)
  CASE(0x10); CASE(0x11); CASE(0x12); CASE(0x13); CASE(0x14); CASE(0x15); CASE(0x16); CASE(0x17)
  CASE(0x18); CASE(0x19); CASE(0x1a); CASE(0x1b); CASE(0x1c); CASE(0x1d); CASE(0x1e); CASE(0x1f)
  CASE(0x20); CASE(0x21); CASE(0x22); CASE(0x23); CASE(0x24); CASE(0x25); CASE(0x26); CASE(0x27)
  CASE(0x28); CASE(0x29); CASE(0x2a); CASE(0x2b); CASE(0x2c); CASE(0x2d); CASE(0x2e); CASE(0x2f)
  CASE(0x30); CASE(0x31); CASE(0x32); CASE(0x33); CASE(0x34); CASE(0x35); CASE(0x36); CASE(0x37)
  CASE(0x38); CASE(0x39); CASE(0x3a); CASE(0x3b); CASE(0x3c); CASE(0x3d); CASE(0x3e); CASE(0x3f)
  CASE(0x40); CASE(0x41); CASE(0x42); CASE(0x43); CASE(0x44); CASE(0x45); CASE(0x46); CASE(0x47)
  CASE(0x48); CASE(0x49); CASE(0x4a); CASE(0x4b); CASE(0x4c); CASE(0x4d); CASE(0x4e); CASE(0x4f)
  CASE(0x50); CASE(0x51); CASE(0x52); CASE(0x53); CASE(0x54); CASE(0x55); CASE(0x56); CASE(0x57)
  CASE(0x58); CASE(0x59); CASE(0x5a); CASE(0x5b); CASE(0x5c); CASE(0x5d); CASE(0x5e); CASE(0x5f)
  CASE(0x60); CASE(0x61); CASE(0x62); CASE(0x63); CASE(0x64); CASE(0x65); CASE(0x66); CASE(0x67)
  CASE(0x68); CASE(0x69); CASE(0x6a); CASE(0x6b); CASE(0x6c); CASE(0x6d); CASE(0x6e); CASE(0x6f)
  CASE(0x70); CASE(0x71); CASE(0x72); CASE(0x73); CASE(0x74); CASE(0x75); CASE(0x76); CASE(0x77)
  CASE(0x78); CASE(0x79); CASE(0x7a); CASE(0x7b); CASE(0x7c); CASE(0x7d); CASE(0x7e); CASE(0x7f)
  CASE(0x80); CASE(0x81); CASE(0x82); CASE(0x83); CASE(0x84); CASE(0x85); CASE(0x86); CASE(0x87)
  CASE(0x88); CASE(0x89); CASE(0x8a); CASE(0x8b); CASE(0x8c); CASE(0x8d); CASE(0x8e); CASE(0x8f)
  CASE(0x90); CASE(0x91); CASE(0x92); CASE(0x93); CASE(0x94); CASE(0x95); CASE(0x96); CASE(0x97)
  CASE(0x98); CASE(0x99); CASE(0x9a); CASE(0x9b); CASE(0x9c); CASE(0x9d); CASE(0x9e); CASE(0x9f)
  CASE(0xa0); CASE(0xa1); CASE(0xa2); CASE(0xa3); CASE(0xa4); CASE(0xa5); CASE(0xa6); CASE(0xa7)
  CASE(0xa8); CASE(0xa9); CASE(0xaa); CASE(0xab); CASE(0xac); CASE(0xad); CASE(0xae); CASE(0xaf)
  CASE(0xb0); CASE(0xb1); CASE(0xb2); CASE(0xb3); CASE(0xb4); CASE(0xb5); CASE(0xb6); CASE(0xb7)
  CASE(0xb8); CASE(0xb9); CASE(0xba); CASE(0xbb); CASE(0xbc); CASE(0xbd); CASE(0xbe); CASE(0xbf)
  CASE(0xc0); CASE(0xc1); CASE(0xc2); CASE(0xc3); CASE(0xc4); CASE(0xc5); CASE(0xc6); CASE(0xc7)
  CASE(0xc8); CASE(0xc9); CASE(0xca); CASE(0xcb); CASE(0xcc); CASE(0xcd); CASE(0xce); CASE(0xcf)
  CASE(0xd0); CASE(0xd1); CASE(0xd2); CASE(0xd3); CASE(0xd4); CASE(0xd5); CASE(0xd6); CASE(0xd7)
  CASE(0xd8); CASE(0xd9); CASE(0xda); CASE(0xdb); CASE(0xdc); CASE(0xdd); CASE(0xde); CASE(0xdf)
  CASE(0xe0); CASE(0xe1); CASE(0xe2); CASE(0xe3); CASE(0xe4); CASE(0xe5); CASE(0xe6); CASE(0xe7)
  CASE(0xe8); CASE(0xe9); CASE(0xea); CASE(0xeb); CASE(0xec); CASE(0xed); CASE(0xee); CASE(0xef)
  CASE(0xf0); CASE(0xf1); CASE(0xf2); CASE(0xf3); CASE(0xf4); CASE(0xf5); CASE(0xf6); CASE(0xf7)
  CASE(0xf8); CASE(0xf9); CASE(0xfa); CASE(0xfb); CASE(0xfc); CASE(0xfd); CASE(0xfe); CASE(0xff)
  default: abort();
  }

#undef CASE

  CAMLreturn(U::toocaml(aout));
}

/* -------------------------------------------------------------------- */
template <auto F, typename U, typename T1, typename T2>
static value bindc(value arg1, value arg2, value imm8) {
  CAMLparam3(arg1, arg2, imm8);
  typename T1::type varg1 = T1::ofocaml(arg1);
  typename T2::type varg2 = T2::ofocaml(arg2);
  const uint8_t vimm8 = (uint8_t) Long_val(imm8);
  typename U::type aout;

#define CASE(I) case I: { aout = F(varg1, varg2, I); break ; }

  switch (vimm8) {
  CASE(0x00); CASE(0x01); CASE(0x02); CASE(0x03); CASE(0x04); CASE(0x05); CASE(0x06); CASE(0x07)
  CASE(0x08); CASE(0x09); CASE(0x0a); CASE(0x0b); CASE(0x0c); CASE(0x0d); CASE(0x0e); CASE(0x0f)
  CASE(0x10); CASE(0x11); CASE(0x12); CASE(0x13); CASE(0x14); CASE(0x15); CASE(0x16); CASE(0x17)
  CASE(0x18); CASE(0x19); CASE(0x1a); CASE(0x1b); CASE(0x1c); CASE(0x1d); CASE(0x1e); CASE(0x1f)
  CASE(0x20); CASE(0x21); CASE(0x22); CASE(0x23); CASE(0x24); CASE(0x25); CASE(0x26); CASE(0x27)
  CASE(0x28); CASE(0x29); CASE(0x2a); CASE(0x2b); CASE(0x2c); CASE(0x2d); CASE(0x2e); CASE(0x2f)
  CASE(0x30); CASE(0x31); CASE(0x32); CASE(0x33); CASE(0x34); CASE(0x35); CASE(0x36); CASE(0x37)
  CASE(0x38); CASE(0x39); CASE(0x3a); CASE(0x3b); CASE(0x3c); CASE(0x3d); CASE(0x3e); CASE(0x3f)
  CASE(0x40); CASE(0x41); CASE(0x42); CASE(0x43); CASE(0x44); CASE(0x45); CASE(0x46); CASE(0x47)
  CASE(0x48); CASE(0x49); CASE(0x4a); CASE(0x4b); CASE(0x4c); CASE(0x4d); CASE(0x4e); CASE(0x4f)
  CASE(0x50); CASE(0x51); CASE(0x52); CASE(0x53); CASE(0x54); CASE(0x55); CASE(0x56); CASE(0x57)
  CASE(0x58); CASE(0x59); CASE(0x5a); CASE(0x5b); CASE(0x5c); CASE(0x5d); CASE(0x5e); CASE(0x5f)
  CASE(0x60); CASE(0x61); CASE(0x62); CASE(0x63); CASE(0x64); CASE(0x65); CASE(0x66); CASE(0x67)
  CASE(0x68); CASE(0x69); CASE(0x6a); CASE(0x6b); CASE(0x6c); CASE(0x6d); CASE(0x6e); CASE(0x6f)
  CASE(0x70); CASE(0x71); CASE(0x72); CASE(0x73); CASE(0x74); CASE(0x75); CASE(0x76); CASE(0x77)
  CASE(0x78); CASE(0x79); CASE(0x7a); CASE(0x7b); CASE(0x7c); CASE(0x7d); CASE(0x7e); CASE(0x7f)
  CASE(0x80); CASE(0x81); CASE(0x82); CASE(0x83); CASE(0x84); CASE(0x85); CASE(0x86); CASE(0x87)
  CASE(0x88); CASE(0x89); CASE(0x8a); CASE(0x8b); CASE(0x8c); CASE(0x8d); CASE(0x8e); CASE(0x8f)
  CASE(0x90); CASE(0x91); CASE(0x92); CASE(0x93); CASE(0x94); CASE(0x95); CASE(0x96); CASE(0x97)
  CASE(0x98); CASE(0x99); CASE(0x9a); CASE(0x9b); CASE(0x9c); CASE(0x9d); CASE(0x9e); CASE(0x9f)
  CASE(0xa0); CASE(0xa1); CASE(0xa2); CASE(0xa3); CASE(0xa4); CASE(0xa5); CASE(0xa6); CASE(0xa7)
  CASE(0xa8); CASE(0xa9); CASE(0xaa); CASE(0xab); CASE(0xac); CASE(0xad); CASE(0xae); CASE(0xaf)
  CASE(0xb0); CASE(0xb1); CASE(0xb2); CASE(0xb3); CASE(0xb4); CASE(0xb5); CASE(0xb6); CASE(0xb7)
  CASE(0xb8); CASE(0xb9); CASE(0xba); CASE(0xbb); CASE(0xbc); CASE(0xbd); CASE(0xbe); CASE(0xbf)
  CASE(0xc0); CASE(0xc1); CASE(0xc2); CASE(0xc3); CASE(0xc4); CASE(0xc5); CASE(0xc6); CASE(0xc7)
  CASE(0xc8); CASE(0xc9); CASE(0xca); CASE(0xcb); CASE(0xcc); CASE(0xcd); CASE(0xce); CASE(0xcf)
  CASE(0xd0); CASE(0xd1); CASE(0xd2); CASE(0xd3); CASE(0xd4); CASE(0xd5); CASE(0xd6); CASE(0xd7)
  CASE(0xd8); CASE(0xd9); CASE(0xda); CASE(0xdb); CASE(0xdc); CASE(0xdd); CASE(0xde); CASE(0xdf)
  CASE(0xe0); CASE(0xe1); CASE(0xe2); CASE(0xe3); CASE(0xe4); CASE(0xe5); CASE(0xe6); CASE(0xe7)
  CASE(0xe8); CASE(0xe9); CASE(0xea); CASE(0xeb); CASE(0xec); CASE(0xed); CASE(0xee); CASE(0xef)
  CASE(0xf0); CASE(0xf1); CASE(0xf2); CASE(0xf3); CASE(0xf4); CASE(0xf5); CASE(0xf6); CASE(0xf7)
  CASE(0xf8); CASE(0xf9); CASE(0xfa); CASE(0xfb); CASE(0xfc); CASE(0xfd); CASE(0xfe); CASE(0xff)
  default: abort();
  }

#undef CASE

  CAMLreturn(U::toocaml(aout));
}

/* -------------------------------------------------------------------- */
# define BIND1(F, U, T)                                    \
CAMLprim value NX(caml_,F)(value a) {                      \
  return bind<SIMD(F), U, T>(a);                           \
}

/* -------------------------------------------------------------------- */
# define BIND2(F, U, T1, T2)                               \
CAMLprim value NX(caml_,F)(value a, value b) {             \
  return bind<SIMD(F), U, T1, T2>(a, b);                   \
}

/* -------------------------------------------------------------------- */
# define BIND3(F, U, T1, T2, T3)                           \
CAMLprim value NX(caml_,F)(value a, value b, value c) {    \
  return bind<SIMD(F), U, T1, T2, T3>(a, b, c);            \
}

/* -------------------------------------------------------------------- */
# define BIND1C(F, U, T)                                   \
CAMLprim value NX(caml_,F)(value a, value imm8) {          \
  return bindc<SIMD(F), U, T>(a, imm8);                    \
}

/* -------------------------------------------------------------------- */
# define BIND2C(F, U, T1, T2)                              \
CAMLprim value NX(caml_,F)(value a, value b, value imm8) { \
  return bindc<SIMD(F), U, T1, T2>(a, b, imm8);            \
}

/* -------------------------------------------------------------------- */
#define BIND_128x2_128(F) BIND2(F, M128i, M128i, M128i)
#define BIND_256x2_256(F) BIND2(F, M256i, M256i, M256i)
#define BIND_256x3_256(F) BIND3(F, M256i, M256i, M256i, M256i)

/* -------------------------------------------------------------------- */
#define BINDC_128x2_128(F) BIND2C(F, M128i, M128i, M128i)
#define BINDC_256x2_256(F) BIND2C(F, M256i, M256i, M256i)

/* -------------------------------------------------------------------- */
extern "C" {
BIND1(mm256_broadcastq_epi64, M256i, M128i);
BIND1(mm256_broadcastd_epi32, M256i, M128i);
BIND1(mm256_broadcastw_epi16, M256i, M128i);
BIND1(mm256_broadcastsi128_si256, M256i, M128i);

BIND_256x2_256(mm256_shuffle_epi8);
BIND1C(mm256_shuffle_epi32, M256i, M256i);

BIND1C(mm256_permute4x64_epi64, M256i, M256i);
BIND_256x2_256(mm256_permutevar8x32_epi32);
BINDC_256x2_256(mm256_permute2x128_si256);

BINDC_128x2_128(mm_blend_epi16);
BINDC_128x2_128(mm_blend_epi32);

BINDC_256x2_256(mm256_blend_epi16);
BINDC_256x2_256(mm256_blend_epi32);

BIND_256x2_256(mm256_packs_epi16)
BIND_256x2_256(mm256_packs_epi32)

BIND_256x2_256(mm256_packus_epi16)
BIND_256x2_256(mm256_packus_epi32)

BIND_128x2_128(mm_and_si128);
BIND_256x2_256(mm256_and_si256);

BIND_128x2_128(mm_andnot_si128);
BIND_256x2_256(mm256_andnot_si256);

BIND_128x2_128(mm_or_si128);
BIND_256x2_256(mm256_or_si256);

BIND_128x2_128(mm_xor_si128);
BIND_256x2_256(mm256_xor_si256);

BIND_256x2_256(mm256_add_epi8);
BIND_256x2_256(mm256_add_epi16);
BIND_256x2_256(mm256_add_epi32);
BIND_256x2_256(mm256_add_epi64);

BIND_256x2_256(mm256_sub_epi8);
BIND_256x2_256(mm256_sub_epi16);
BIND_256x2_256(mm256_sub_epi32);
BIND_256x2_256(mm256_sub_epi64);

BIND_256x2_256(mm256_maddubs_epi16);
BIND_256x2_256(mm256_madd_epi16);

BIND_256x2_256(mm256_mullo_epi16);
BIND_256x2_256(mm256_mulhi_epi16);
BIND_256x2_256(mm256_mulhi_epu16);

BIND_256x2_256(mm256_mulhrs_epi16);

BIND_256x2_256(mm256_cmpgt_epi8);
BIND_256x2_256(mm256_cmpgt_epi16);
BIND_256x2_256(mm256_cmpgt_epi32);
BIND_256x2_256(mm256_cmpgt_epi64);

BIND2(mm256_srl_epi16, M256i, M256i, M128i);
BIND2(mm256_srl_epi32, M256i, M256i, M128i);
BIND2(mm256_srl_epi64, M256i, M256i, M128i);

BIND2(mm256_sra_epi16, M256i, M256i, M128i);
BIND2(mm256_sra_epi32, M256i, M256i, M128i);

BIND2(mm256_sll_epi16, M256i, M256i, M128i);
BIND2(mm256_sll_epi32, M256i, M256i, M128i);
BIND2(mm256_sll_epi64, M256i, M256i, M128i);

BIND2(mm256_sllv_epi32, M256i, M256i, M256i);
BIND2(mm256_sllv_epi64, M256i, M256i, M256i);

BIND2(mm256_srlv_epi32, M256i, M256i, M256i);
BIND2(mm256_srlv_epi64, M256i, M256i, M256i);

BIND1C(mm_bslli_si128, M128i, M128i);
BIND1C(mm_bsrli_si128, M128i, M128i);
BIND1C(mm256_bslli_epi128, M256i, M256i);
BIND1C(mm256_bsrli_epi128, M256i, M256i);

BIND_128x2_128(mm_unpacklo_epi8);
BIND_128x2_128(mm_unpacklo_epi16);
BIND_128x2_128(mm_unpacklo_epi32);
BIND_128x2_128(mm_unpacklo_epi64);

BIND_128x2_128(mm_unpackhi_epi8);
BIND_128x2_128(mm_unpackhi_epi16);
BIND_128x2_128(mm_unpackhi_epi32);
BIND_128x2_128(mm_unpackhi_epi64);

BIND_256x2_256(mm256_unpacklo_epi8);
BIND_256x2_256(mm256_unpacklo_epi16);
BIND_256x2_256(mm256_unpacklo_epi32);
BIND_256x2_256(mm256_unpacklo_epi64);

BIND_256x2_256(mm256_unpackhi_epi8);
BIND_256x2_256(mm256_unpackhi_epi16);
BIND_256x2_256(mm256_unpackhi_epi32);
BIND_256x2_256(mm256_unpackhi_epi64);
}
