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
#ifndef AVX2_BINDC_CASES
#define AVX2_BINDC_CASES 256
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
extern "C" {
#include "avx2_runtime_bindings.inc"
}
