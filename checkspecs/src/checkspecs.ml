(* ======================================================================== *)
open Batteries

(* ------------------------------------------------------------------------ *)
module A = Avx2
module S = Lospecs

(* ======================================================================== *)
type _ primty =
  | M256  : A.m256 primty
  | M128  : A.m128 primty
  | Imm8  : int primty

(* ------------------------------------------------------------------------ *)
type primval =
  | PrimVal : 'a primty * 'a -> primval

(* ------------------------------------------------------------------------ *)
type extend = ENone | EZero

(* ------------------------------------------------------------------------ *)
type (_, _) primsig =
  | Ret : 'r primty -> ('r, 'r) primsig
  | Arg : 'a primty * extend * ('b, 'r) primsig -> ('a -> 'b, 'r) primsig

(* ------------------------------------------------------------------------ *)
type prim =
  | Prim : ('a, 'r) primsig * 'a -> prim

(* ------------------------------------------------------------------------ *)
let mkarg
  : type a b r . ?extend:extend -> a primty -> (b, r) primsig -> (a -> b, r) primsig
  = fun ?(extend = ENone) a r -> Arg (a, extend, r)

(* ------------------------------------------------------------------------ *)
let mkret
  : type r . r primty -> (r, r) primsig
  = fun r -> Ret r

(* ------------------------------------------------------------------------ *)
let prim1 ?e a r f =
  Prim (mkarg ?extend:e a (mkret r), f)

(* ------------------------------------------------------------------------ *)
let prim2 ?e1 ?e2 (a1, a2) r f =
  Prim (mkarg ?extend:e1 a1 (mkarg ?extend:e2 a2 (mkret r)), f)

(* ------------------------------------------------------------------------ *)
let prim3 ?e1 ?e2 ?e3 (a1, a2, a3) r f =
  Prim (mkarg ?extend:e1 a1 (mkarg ?extend:e2 a2 (mkarg ?extend:e3 a3 (mkret r))), f)

(* ======================================================================== *)
type test = {
  name : string;
  prim : prim;
}

(* ------------------------------------------------------------------------ *)
let tests : test list = [
  { name = "VPBROADCAST_2u128"
  ; prim = (prim1 M128 M256 Avx2.mm256_broadcastsi128_si256); }
  ;
  { name = "VPBROADCAST_4u64"
  ; prim = (prim1 ~e:EZero M128 M256 Avx2.mm256_broadcastq_epi64); }
  ;
  { name = "VPBROADCAST_8u32"
  ; prim = (prim1 ~e:EZero M128 M256 Avx2.mm256_broadcastd_epi32); }
  ;
  { name = "VPBROADCAST_16u16"
  ; prim = (prim1 ~e:EZero M128 M256 Avx2.mm256_broadcastw_epi16); }
  ;
  { name = "VPSHUFD_256"
  ; prim = (prim2 (M256, Imm8) M256 Avx2.mm256_shuffle_epi32); }
  ;
  { name = "VPSHUFB_256"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_shuffle_epi8); }
  ;
  { name = "VPERMQ"
  ; prim = (prim2 (M256, Imm8) M256 Avx2.mm256_permute4x64_epi64); }
  ;
  { name = "VPERMD"
  ; prim = (prim2 (M256, M256) M256 (fun w idx -> Avx2.mm256_permutevar8x32_epi32 idx w)); }
  ;
  { name = "VPAND_128"
  ; prim = (prim2 (M128, M128) M128 Avx2.mm_and_si128); }
  ;
  { name = "VPAND_256"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_and_si256); }
  ;
  { name = "VPANDN_128"
  ; prim = (prim2 (M128, M128) M128 Avx2.mm_andnot_si128); }
  ;
  { name = "VPANDN_256"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_andnot_si256); }
  ;
  { name = "VPOR_128"
  ; prim = (prim2 (M128, M128) M128 Avx2.mm_or_si128); }
  ;
  { name = "VPOR_256"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_or_si256); }
  ;
  { name = "VPXOR_128"
  ; prim = (prim2 (M128, M128) M128 Avx2.mm_xor_si128); }
  ;
  { name = "VPXOR_256"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_xor_si256); }
  ;
  { name = "VPADD_4u64"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_add_epi64); }
  ;
  { name = "VPADD_8u32"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_add_epi32); }
  ;
  { name = "VPADD_16u16"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_add_epi16); }
  ;
  { name = "VPADD_32u8"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_add_epi8); }
  ;
  { name = "VPSUB_4u64"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_sub_epi64); }
  ;
  { name = "VPSUB_8u32"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_sub_epi32); }
  ;
  { name = "VPSUB_16u16"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_sub_epi16); }
  ;
  { name = "VPSUB_32u8"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_sub_epi8); }
  ;
  { name = "VPMULL_16u16"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_mullo_epi16); }
  ;
  { name = "VPMULH_16u16"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_mulhi_epi16); }
  ;
  { name = "VPMULHU_16u16"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_mulhi_epu16); }
  ;
  { name = "VPMULHRS_16u16"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_mulhrs_epi16); }
  ;
  { name = "VPCMPGT_8u32"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_cmpgt_epi32); }
  ;
  { name = "VPCMPGT_16u16"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_cmpgt_epi16); }
  ;
  { name = "VPCMPGT_32u8"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_cmpgt_epi8); }
  ;
  { name = "VPSRL_4u64"
  ; prim = (prim2 (M256, M128) M256 Avx2.mm256_srl_epi64); }
  ;
  { name = "VPSRL_8u32"
  ; prim = (prim2 (M256, M128) M256 Avx2.mm256_srl_epi32); }
  ;
  { name = "VPSRL_16u16"
  ; prim = (prim2 (M256, M128) M256 Avx2.mm256_srl_epi16); }
  ;
  { name = "VPSRA_8u32"
  ; prim = (prim2 (M256, M128) M256 Avx2.mm256_sra_epi32); }
  ;
  { name = "VPSRA_16u16"
  ; prim = (prim2 (M256, M128) M256 Avx2.mm256_sra_epi16); }
  ;
  { name = "VPSLL_4u64"
  ; prim = (prim2 (M256, M128) M256 Avx2.mm256_sll_epi64); }
  ;
  { name = "VPSLL_8u32"
  ; prim = (prim2 (M256, M128) M256 Avx2.mm256_sll_epi32); }
  ;
  { name = "VPSLL_16u16"
  ; prim = (prim2 (M256, M128) M256 Avx2.mm256_sll_epi16); }
  ;
  { name = "VPSLLV_4u64"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_sllv_epi64); }
  ;
  { name = "VPSLLV_8u32"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_sllv_epi32); }
  ;
  { name = "VPSRLV_4u64"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_srlv_epi64); }
  ;
  { name = "VPSRLV_8u32"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_srlv_epi32); }
  ;
  { name = "VPUNPCKL_4u64"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_unpacklo_epi64); }
  ;
  { name = "VPUNPCKL_8u32"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_unpacklo_epi32); }
  ;
  { name = "VPUNPCKL_16u16"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_unpacklo_epi16); }
  ;
  { name = "VPUNPCKL_32u8"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_unpacklo_epi8); }
  ;
  { name = "VPUNPCKH_4u64"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_unpackhi_epi64); }
  ;
  { name = "VPUNPCKH_8u32"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_unpackhi_epi32); }
  ;
  { name = "VPUNPCKH_16u16"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_unpackhi_epi16); }
  ;
  { name = "VPUNPCKH_32u8"
  ; prim = (prim2 (M256, M256) M256 Avx2.mm256_unpackhi_epi8); }
  ;
]

(* ======================================================================== *)
let tests =
  tests
  |> List.to_seq
  |> Seq.map (fun t -> (t.name, t))
  |> BatMap.of_seq

(* ------------------------------------------------------------------------ *)
type testerror =
  | MissingTest of string
  | InvalidSignature
  | InvalidOutput of { inputs: bytes list; expected: bytes; got: bytes }

exception TestError of testerror

(* ------------------------------------------------------------------------ *)
let check_ty_compatible 
  : type a . ?extend:extend -> a primty -> S.Ast.atype -> unit
  = fun ?(extend : extend = ENone) (a : _ primty) (ty : S.Ast.atype)  ->

    match ty with
    | `W tysize -> begin
      let asize = match a with M256 -> 256 | M128 -> 128 | Imm8 -> 8 in
      match extend with
      | ENone when tysize = asize -> ()
      | EZero when tysize <= asize -> ()
      | _ -> raise (TestError InvalidSignature)
    end
    | _ -> raise (TestError InvalidSignature)

(* ------------------------------------------------------------------------ *)
let check_signature (t : test) (c : S.Ast.adef) =
  let rec check_signature
    : type a r . (a, r) primsig -> S.Ast.aargs -> unit
    = fun (sig_ : (_, _) primsig) (arguments : S.Ast.aargs) ->
      match sig_, arguments with
      | Ret retty, [] ->
        check_ty_compatible retty (c.rettype :> S.Ast.atype)
      | Arg (aty, extend, asty), (_, argty) :: args ->
        check_ty_compatible ~extend aty (argty :> S.Ast.atype);
        check_signature asty args
      | _, _ ->
        raise (TestError InvalidSignature)
  in

  let Prim (sig_, _) = t.prim in
  check_signature sig_ c.arguments

(* ------------------------------------------------------------------------ *)
let generate_input
  : type a . a primty -> a
  = fun (aty : a primty) : a ->

  match aty with
  | M256 -> A.M256.random ()
  | M128 -> A.M128.random ()
  | Imm8 -> Random.int_in_range ~min:0 ~max:255

(* ------------------------------------------------------------------------ *)
let generate_test_vector (t : test) =
  let rec generate
    : type a r . (a, r) primsig -> a -> primval list * primval
    = fun (sig_ : (a, r) primsig) (refv : a) ->

      match sig_ with
      | Ret ty ->
        ([], PrimVal (ty, refv))
      | Arg (aty, _, rty) ->
        let input = generate_input aty in
        let (inputs, refv) = generate rty (refv input) in
        (PrimVal (aty, input) :: inputs, refv)
  in

  let Prim (sig_, refv) = t.prim in
  generate sig_ refv

(* ------------------------------------------------------------------------ *)
let bytes_of_primval (p : primval) : bytes =
  match p with
  | PrimVal (M256, m256) ->
    A.M256.to_bytes ~endianess:`Little m256
  | PrimVal (M128, m128) ->
    A.M128.to_bytes ~endianess:`Little m128
  | PrimVal (Imm8, v) ->
    Bytes.of_string (String.of_char (Char.chr v))

(* ======================================================================== *)
let bar (name : string) (total : int) =
  let open Progress.Line in
  list [
      spinner ~color:(Progress.Color.ansi `green) ()
    ; rpad (max 20 (String.length name)) (const name)
    ; bar total
    ; lpad (2 * 7 + 1) (count_to total)
  ]

(* ------------------------------------------------------------------------ *)
let non_interactive =
  (not Unix.(isatty stderr))
  || Option.is_some (Sys.getenv_opt "CI")

(* ------------------------------------------------------------------------ *)
let log_non_tty (msg : string) =
  if non_interactive then
    Format.eprintf "%s@." msg

(* ======================================================================== *)
let evaluate_test ~(ntests : int) (t : test) (c : S.Ast.adef) =
  check_signature t c;

  let reg_of_aword (name : int) (`W size : S.Ast.aword) =
    S.Circuit.reg ~name ~size in

  let regs = List.mapi reg_of_aword (List.map snd c.arguments) in
  let circuit = S.Circuit_spec.circuit_of_specification regs c in

  let bar = bar t.name ntests in

  let dotest () =
    let inputs, output = generate_test_vector t in
    let output = bytes_of_primval output in
    let inputs = List.map bytes_of_primval inputs in
    let inputs = Array.of_list inputs in

    let env ((n, i) : S.Aig.var) = S.Aig.get_bit inputs.(n) i in
    let cout = Array.map (S.Aig.eval env) circuit in
    let cout = S.Circuit.bytes_of_bools cout in

    if cout <> output then
      raise (TestError (InvalidOutput {
        inputs   = Array.to_list inputs;
        expected = output;
        got      = cout;
      }))
  in

  if non_interactive then
    for _ = 1 to ntests do
      dotest ()
    done
  else
    Progress.with_reporter bar (fun f ->
      for _ = 1 to ntests do
        dotest (); f 1
      done
    )

(* ------------------------------------------------------------------------ *)
let evaluate_test ~(ntests : int) ((name, adef) : string * S.Ast.adef) : bool =
  log_non_tty (Format.sprintf "[checkspecs] start %s" name);
  try
    BatMap.find_opt name tests
    |> fun x -> BatOption.get_exn x (TestError (MissingTest name))
    |> (fun test -> evaluate_test ~ntests test adef);
    log_non_tty (Format.sprintf "[checkspecs] stop %s (ok)" name);
    true

  with
  | TestError e ->
    begin
      match e with
      | InvalidOutput e ->
        let width = 16 in
        List.iteri (fun i b ->
          Format.eprintf "%-*s: 0x%s@."
            width (Format.sprintf "input #%.2d" i)
            (Hex.of_bytes b |> Hex.show)
        ) e.inputs;

        Format.eprintf "%-*s: 0x%s@." width "got" (Hex.of_bytes e.got |> Hex.show);
        Format.eprintf "%-*s: 0x%s@." width "expected" (Hex.of_bytes e.expected |> Hex.show)

      | MissingTest _ ->
        Format.eprintf "Missing test for `%s'@." name

      | InvalidSignature ->
        Format.eprintf "Mismatch signature for `%s'@." name
    end;
    log_non_tty (Format.sprintf "[checkspecs] stop %s (failed)" name);
    false

(* ======================================================================== *)
module CLI = struct
  (* ---------------------------------------------------------------------- *)
  let positive_int_conv : int Cmdliner.Arg.conv =
    let parse (s : string) =
      try
        let n = int_of_string s in
        if n > 0 then
          Ok n
        else
          Error (`Msg "must be > 0")
      with Failure _ ->
        Error (`Msg "must be an integer")
    in
    Cmdliner.Arg.conv (parse, Format.pp_print_int)

  (* ---------------------------------------------------------------------- *)
  let regexp_conv : Str.regexp Cmdliner.Arg.conv =
    let parse (s : string) =
      try Ok (Str.regexp s) with
      | Failure msg -> Error (`Msg ("invalid regexp: " ^ msg))
    in
    let print fmt _ =
      Format.pp_print_string fmt "<regexp>"
    in
    Cmdliner.Arg.conv (parse, print)
end

(* ------------------------------------------------------------------------ *)
type cli = {
  ntests : int;
  filter : Str.regexp option;
  fail_on_simde : bool;
  spec : string;
}

(* ------------------------------------------------------------------------ *)
let cli_term : cli Cmdliner.Term.t =
  let open Cmdliner in

  let ntests =
    Arg.(
      required
      & opt (some CLI.positive_int_conv) None
      & info ["n"] ~docv:"NUMTESTS" ~doc:"Number of tests to run per spec (must be > 0).") in
  let filter =
    Arg.(
      value
      & opt (some CLI.regexp_conv) None
      & info ["filter"] ~docv:"REGEXP" ~doc:"Regexp used to filter spec names.") in
  let fail_on_simde =
    Arg.(
      value
      & flag
      & info ["fail-on-simde"] ~doc:"Fail immediately when SIMD implementation is SIMDe.") in
  let spec =
    Arg.(
      required
      & pos 0 (some string) None
      & info [] ~docv:"FILENAME" ~doc:"Specification filename.") in

  Term.(const (fun ntests filter fail_on_simde spec ->
    { ntests; filter; fail_on_simde; spec }
  ) $ ntests $ filter $ fail_on_simde $ spec)

(* ------------------------------------------------------------------------ *)
let main (cli : cli) =
  if cli.fail_on_simde && A.using_simde () then (
    Format.eprintf "SIMDe is enabled; refusing to run with --fail-on-simde.@.";
    exit 1
  );

  let specs  = File.with_file_in cli.spec (S.Io.parse cli.spec) in
  let specs  = S.Typing.tt_program S.Typing.Env.empty specs in
  let specs  =
    match cli.filter with
    | None -> specs
    | Some re ->
      let filter (name, _) =
        try
          ignore (Str.search_forward re name 0 : int);
          true
        with Not_found -> false in
      List.filter filter specs
  in
  let status =
    specs
    |> List.to_seq
    |> Seq.map (evaluate_test ~ntests:cli.ntests)
    |> Seq.fold_left (&&) true
  in

  if not status then
    exit 1

(* ------------------------------------------------------------------------ *)
let () =
  let open Cmdliner in
  let doc = "Run spec tests against reference implementations / hardware." in
  let cmd = Cmd.v (Cmd.info "main" ~doc) Term.(const main $ cli_term) in
  exit (Cmd.eval cmd)
