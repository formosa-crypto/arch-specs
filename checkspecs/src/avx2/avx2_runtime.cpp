/* ==================================================================== */
#include <stdlib.h>
#include <stdint.h>

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
# define BIND1(F, U, T)                                \
CAMLprim value NX(caml_,F)(value a) {                     \
  return bind<SIMD(F), U, T>(a);                          \
}

/* -------------------------------------------------------------------- */
# define BIND2(F, U, T1, T2)                           \
CAMLprim value NX(caml_,F)(value a, value b) {            \
  return bind<SIMD(F), U, T1, T2>(a, b);                  \
}

/* -------------------------------------------------------------------- */
# define BIND3(F, U, T1, T2, T3)                       \
CAMLprim value NX(caml_,F)(value a, value b, value c) {   \
  return bind<SIMD(F), U, T1, T2, T3>(a, b, c);           \
}

#define BIND_256x2_256(F) BIND2(F, M256i, M256i, M256i)
#define BIND_256x3_256(F) BIND3(F, M256i, M256i, M256i, M256i)
#define BIND_128x2_128(F) BIND2(F, M128i, M128i, M128i)

extern "C" {
BIND1(mm256_broadcastq_epi64, M256i, M128i);
BIND1(mm256_broadcastd_epi32, M256i, M128i);
BIND1(mm256_broadcastw_epi16, M256i, M128i);
BIND1(mm256_broadcastsi128_si256, M256i, M128i);

BIND_256x2_256(mm256_permutevar8x32_epi32);

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

BIND_256x2_256(mm256_mullo_epi16);
BIND_256x2_256(mm256_mulhi_epi16);
BIND_256x2_256(mm256_mulhi_epu16);

BIND_256x2_256(mm256_mulhrs_epi16);

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

BIND2(mm256_unpacklo_epi8, M256i, M256i, M256i);
BIND2(mm256_unpacklo_epi16, M256i, M256i, M256i);
BIND2(mm256_unpacklo_epi32, M256i, M256i, M256i);
BIND2(mm256_unpacklo_epi64, M256i, M256i, M256i);

BIND2(mm256_unpackhi_epi8, M256i, M256i, M256i);
BIND2(mm256_unpackhi_epi16, M256i, M256i, M256i);
BIND2(mm256_unpackhi_epi32, M256i, M256i, M256i);
BIND2(mm256_unpackhi_epi64, M256i, M256i, M256i);
}
