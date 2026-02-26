/* ==================================================================== */
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <array>
#include <cstddef>
#include <utility>

/* -------------------------------------------------------------------- */
#define NX(A, B) A ## B

/* -------------------------------------------------------------------- */
#ifdef SIMDe
# include <simde/x86/avx.h>
# include <simde/x86/avx2.h>
# define SIMD(B) NX(simde_, B)
#else
# include <immintrin.h>
# define SIMD(B) NX(_, B)
#endif

/* -------------------------------------------------------------------- */
typedef SIMD(_m128)  m128;
typedef SIMD(_m128i) m128i;

typedef SIMD(_m256)  m256;
typedef SIMD(_m256i) m256i;

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
struct M128 {
  typedef m128 type;

  static inline type ofocaml(value v) {
    return SIMD(mm_castsi128_ps)(w128_of_value(v));
  }

  static inline value toocaml(type v) {
    return value_of_w128(SIMD(mm_castps_si128)(v));
  }
};

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
struct M256 {
  typedef m256 type;

  static inline type ofocaml(value v) {
    return SIMD(mm256_castsi256_ps)(w256_of_value(v));
  }

  static inline value toocaml(type v) {
    return value_of_w256(SIMD(mm256_castps_si256)(v));
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
struct Int8 {
  typedef uint8_t type;

  static inline type ofocaml(value v) {
    return Int_val(v);
  }

  static inline value toocaml(type v) {
    return Val_int(v);
  }
};

/* -------------------------------------------------------------------- */
struct Int16 {
  typedef uint16_t type;

  static inline type ofocaml(value v) {
    return Int_val(v);
  }

  static inline value toocaml(type v) {
    return Val_int(v);
  }
};

/* -------------------------------------------------------------------- */
struct Int32 {
  typedef uint32_t type;

  static inline type ofocaml(value v) {
    return Int32_val(v);
  }

  static inline value toocaml(type v) {
    return caml_copy_int32(v);
  }
};

/* -------------------------------------------------------------------- */
struct Int64 {
  typedef uint64_t type;

  static inline type ofocaml(value v) {
    return Int64_val(v);
  }

  static inline value toocaml(type v) {
    return caml_copy_int64(v);
  }
};

/* -------------------------------------------------------------------- */
template <auto F, typename U, typename... Ts, size_t... Is>
static inline value bind_apply(value* args, std::index_sequence<Is...>) {
  return U::toocaml(F(Ts::ofocaml(args[Is])...));
}

/* -------------------------------------------------------------------- */
template <auto F, typename U, typename... Ts>
static value bind(value* args) {
  static_assert(sizeof...(Ts) > 0, "bind expects at least one argument");
  CAMLparamN(args, sizeof...(Ts));
  CAMLreturn((bind_apply<F, U, Ts...>(args, std::index_sequence_for<Ts...>{})));
}

/* -------------------------------------------------------------------- */
template <auto F, typename U, uint8_t I, typename... Ts, size_t... Is>
static inline typename U::type bindc_apply_imm(value* args, std::index_sequence<Is...>) {
  return F(Ts::ofocaml(args[Is])..., I);
}

/* -------------------------------------------------------------------- */
template <auto F, typename U, uint8_t I, typename... Ts>
static inline typename U::type bindc_entry(value* args) {
  return bindc_apply_imm<F, U, I, Ts...>(args, std::index_sequence_for<Ts...>{});
}

/* -------------------------------------------------------------------- */
template <auto F, typename U, typename... Ts, size_t... Is>
static inline std::array<typename U::type (*)(value*), sizeof...(Is)> make_bindc_table(std::index_sequence<Is...>) {
  return { (+[](value* args) -> typename U::type {
    return bindc_entry<F, U, static_cast<uint8_t>(Is), Ts...>(args);
  })... };
}

/* -------------------------------------------------------------------- */
template <auto F, typename U, size_t N, typename... Ts>
static value bindc(value* args, value imm8) {
  static_assert(N > 0, "bindc table size N must be at least 1");
  static_assert(N <= 256, "bindc table size N must be at most 256");
  CAMLparamN(args, sizeof...(Ts));
  CAMLxparam1(imm8);

  static const auto table = make_bindc_table<F, U, Ts...>(std::make_index_sequence<N>{});
  const unsigned long vimm8 = (unsigned long) Int_val(imm8);
  if (vimm8 >= N) {
    abort();
  }
  typename U::type aout = table[vimm8](args);

  CAMLreturn(U::toocaml(aout));
}

/* -------------------------------------------------------------------- */
# define BIND1(F, U, T)                                    \
CAMLprim value NX(caml_,F)(value a) {                      \
  value args[] = {a};                                      \
  return bind<SIMD(F), U, T>(args);                        \
}

/* -------------------------------------------------------------------- */
# define BIND2(F, U, T1, T2)                               \
CAMLprim value NX(caml_,F)(value a, value b) {             \
  value args[] = {a, b};                                   \
  return bind<SIMD(F), U, T1, T2>(args);                   \
}

/* -------------------------------------------------------------------- */
# define BIND3(F, U, T1, T2, T3)                           \
CAMLprim value NX(caml_,F)(value a, value b, value c) {    \
  value args[] = {a, b, c};                                \
  return bind<SIMD(F), U, T1, T2, T3>(args);               \
}

/* -------------------------------------------------------------------- */
# define BIND1C_N(F, U, T, N)                              \
CAMLprim value NX(caml_,F)(value a, value imm8) {          \
  value args[] = {a};                                      \
  return bindc<SIMD(F), U, N, T>(args, imm8);              \
}

/* -------------------------------------------------------------------- */
# define BIND2C_N(F, U, T1, T2, N)                         \
CAMLprim value NX(caml_,F)(value a, value b, value imm8) { \
  value args[] = {a, b};                                   \
  return bindc<SIMD(F), U, N, T1, T2>(args, imm8);         \
}

/* -------------------------------------------------------------------- */
# define BIND1C(F, U, T) BIND1C_N(F, U, T, 256)

/* -------------------------------------------------------------------- */
# define BIND2C(F, U, T1, T2) BIND2C_N(F, U, T1, T2, 256)

/* -------------------------------------------------------------------- */
#define BIND_128x2_128(F) BIND2(F, M128i, M128i, M128i)
#define BIND_128x3_128(F) BIND3(F, M128i, M128i, M128i, M128i)

#define BIND_256x2_256(F) BIND2(F, M256i, M256i, M256i)
#define BIND_256x3_256(F) BIND3(F, M256i, M256i, M256i, M256i)

/* -------------------------------------------------------------------- */
#define BINDC_128x2_128(F) BIND2C(F, M128i, M128i, M128i)
#define BINDC_256x2_256(F) BIND2C(F, M256i, M256i, M256i)

/* -------------------------------------------------------------------- */
extern "C" {
BIND1(mm_moveldup_ps, M128, M128);
BIND1(mm256_moveldup_ps, M256, M256);

BIND1(mm_movemask_epi8, Int32, M128i);
BIND1(mm256_movemask_epi8, Int32, M256i);

BIND1(mm256_broadcastq_epi64, M256i, M128i);
BIND1(mm256_broadcastd_epi32, M256i, M128i);
BIND1(mm256_broadcastw_epi16, M256i, M128i);
BIND1(mm256_broadcastsi128_si256, M256i, M128i);

BIND2C_N(mm_insert_epi8 , M128i, M128i,  Int8, 16);
BIND2C_N(mm_insert_epi16, M128i, M128i, Int16,  8);
BIND2C_N(mm_insert_epi32, M128i, M128i, Int32,  4);
BIND2C_N(mm_insert_epi64, M128i, M128i, Int64,  2);

BIND2C_N(mm256_inserti128_si256, M256i, M256i, M128i, 2);

BIND1C_N(mm_extract_epi8 ,  Int8, M128i, 16);
BIND1C_N(mm_extract_epi16, Int16, M128i,  8);
BIND1C_N(mm_extract_epi32, Int32, M128i,  4);
BIND1C_N(mm_extract_epi64, Int64, M128i,  2);

BIND1C_N(mm256_extracti128_si256, M128i, M256i, 2);

BIND_128x2_128(mm_shuffle_epi8);
BIND1C(mm_shuffle_epi32, M128i, M128i);

BIND_256x2_256(mm256_shuffle_epi8);
BIND1C(mm256_shuffle_epi32, M256i, M256i);

BIND1C(mm256_permute4x64_epi64, M256i, M256i);
BIND_256x2_256(mm256_permutevar8x32_epi32);
BINDC_256x2_256(mm256_permute2x128_si256);

BIND_128x3_128(mm_blendv_epi8)
BIND_256x3_256(mm256_blendv_epi8)

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
