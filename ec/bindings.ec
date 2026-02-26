require import AllCore List QFABV.

from Jasmin require import JWord JModel_x86.

bind bitstring W256.w2bits W256.bits2w W256.to_uint W256.to_sint W256.of_int W256.t 256.
realize size_tolist by admit.
realize tolistP     by admit.
realize oflistP     by admit. 
realize ofintP      by admit.
realize tosintP     by admit.
realize touintP     by admit.
realize gt0_size    by admit.

bind bitstring W128.w2bits W128.bits2w W128.to_uint W128.to_sint W128.of_int W128.t 128.
realize size_tolist by admit.
realize tolistP     by admit.
realize oflistP     by admit. 
realize ofintP      by admit.
realize tosintP     by admit.
realize touintP     by admit.
realize gt0_size    by admit.

bind bitstring W64.w2bits W64.bits2w W64.to_uint W64.to_sint W64.of_int W64.t 64.
realize size_tolist by admit.
realize tolistP     by admit.
realize oflistP     by admit. 
realize ofintP      by admit.
realize tosintP     by admit.
realize touintP     by admit.
realize gt0_size    by admit.

bind bitstring W32.w2bits W32.bits2w W32.to_uint W32.to_sint W32.of_int W32.t 32.
realize size_tolist by admit.
realize tolistP     by admit.
realize oflistP     by admit. 
realize ofintP      by admit.
realize tosintP     by admit.
realize touintP     by admit.
realize gt0_size    by admit.

bind bitstring W16.w2bits W16.bits2w W16.to_uint W16.to_sint W16.of_int W16.t 16.
realize size_tolist by admit.
realize tolistP     by admit.
realize oflistP     by admit. 
realize ofintP      by admit.
realize tosintP     by admit.
realize touintP     by admit.
realize gt0_size    by admit.

bind bitstring W8.w2bits W8.bits2w W8.to_uint W8.to_sint W8.of_int W8.t 8.
realize size_tolist by admit.
realize tolistP     by admit.
realize oflistP     by admit. 
realize ofintP      by admit.
realize tosintP     by admit.
realize touintP     by admit.
realize gt0_size    by admit.

bind circuit
    VMOVSLDUP_128     <- "VMOVSLDUP_128",
    VMOVSLDUP_256     <- "VMOVSLDUP_256",

    MOVEMASK_16u8     <- "VPMOVMSKB_128",
    MOVEMASK_32u8     <- "VPMOVMSKB_256",

    VPINSR_16u8       <- "VPINSR_16u8",
    VPINSR_8u16       <- "VPINSR_8u16",
    VPINSR_4u32       <- "VPINSR_4u32",
    VPINSR_2u64       <- "VPINSR_2u64",
    VINSERTI128       <- "VINSERTI128",

    W16u8.VPEXTR_8    <- "VPEXTR_16u8",
    W8u16.VPEXTR_16   <- "VPEXTR_8u16",
    W4u32.VPEXTR_32   <- "VPEXTR_4u32",
    W2u64.VPEXTR_64   <- "VPEXTR_2u64",
    VEXTRACTI128      <- "VEXTRACTI128",

    VPBROADCAST_4u64  <- "VPBROADCAST_4u64",
    VPBROADCAST_8u32  <- "VPBROADCAST_8u32",
    VPBROADCAST_16u16 <- "VPBROADCAST_16u16",
    VPBROADCAST_2u128 <- "VPBROADCAST_2u128",

    VPERMQ            <- "VPERMQ",
    VPERMD            <- "VPERMD",
    VPERM2I128        <- "VPERM2I128",

    VPSHUFD_128       <- "VPSHUFD_128",
    VPSHUFB_128       <- "VPSHUFB_128",

    VPSHUFD_256       <- "VPSHUFD_256",
    VPSHUFB_256       <- "VPSHUFB_256",

    VPBLEND_4u32      <- "VPBLEND_4u32",
    VPBLEND_8u16      <- "VPBLEND_8u16",

    VPBLEND_8u32      <- "VPBLEND_8u32",
    VPBLEND_16u16     <- "VPBLEND_16u16",

    BLENDV_16u8       <- "VPBLENDVB_128",
    VPBLENDD_128      <- "VPBLEND_4u32",
    VPBLENDW_128      <- "VPBLEND_8u16",

    BLENDV_32u8       <- "VPBLENDVB_256",
    VPBLENDD_256      <- "VPBLEND_8u32",
    VPBLENDW_256      <- "VPBLEND_16u16",

    VPACKSS_16u16     <- "VPACKSS_16u16",
    VPACKSS_8u32      <- "VPACKSS_8u32",

    VPACKUS_16u16     <- "VPACKUS_16u16",
    VPACKUS_8u32      <- "VPACKUS_8u32",

    VPAND_256         <- "VPAND_256",
    VPANDN_256        <- "VPANDN_256",
    VPOR_256          <- "VPOR_256",
    VPXOR_256         <- "VPXOR_256",

    VPAND_128         <- "VPAND_128",
    VPANDN_128        <- "VPANDN_128",
    VPOR_128          <- "VPOR_128",
    VPXOR_128         <- "VPXOR_128",

    VPADD_4u64        <- "VPADD_4u64",
    VPADD_8u32        <- "VPADD_8u32",
    VPADD_16u16       <- "VPADD_16u16",
    VPADD_32u8        <- "VPADD_32u8",

    VPSUB_4u64        <- "VPSUB_4u64",
    VPSUB_8u32        <- "VPSUB_8u32",
    VPSUB_16u16       <- "VPSUB_16u16",
    VPSUB_32u8        <- "VPSUB_32u8",

    VPMULL_16u16      <- "VPMULL_16u16",
    VPMULH_16u16      <- "VPMULH_16u16",
    VPMULHU_16u16     <- "VPMULHU_16u16",
    VPMULHRS_16u16    <- "VPMULHRS_16u16",

    VPMADDUBSW_256    <- "VPMADDUBSW_256",
    VPMADDWD_256      <- "VPMADDWD_256",

    VPCMPGT_8u32      <- "VPCMPGT_8u32",
    VPCMPGT_16u16     <- "VPCMPGT_16u16",
    VPCMPGT_32u8      <- "VPCMPGT_32u8",

    VPSRA_8u32        <- "VPSRA_8u32",
    VPSRA_16u16       <- "VPSRA_16u16",

    VPSRL_4u64        <- "VPSRL_4u64",
    VPSRL_8u32        <- "VPSRL_8u32",
    VPSRL_16u16       <- "VPSRL_16u16",

    VPSLL_4u64        <- "VPSLL_4u64",
    VPSLL_8u32        <- "VPSLL_8u32",
    VPSLL_16u16       <- "VPSLL_16u16",

    VPSLLV_4u64       <- "VPSLLV_4u64",
    VPSLLV_8u32       <- "VPSLLV_8u32",

    VPSRLV_4u64       <- "VPSRLV_4u64",
    VPSRLV_8u32       <- "VPSRLV_8u32",

    VPSLLDQ_128       <- "VPSLLDQ_128",
    VPSRLDQ_128       <- "VPSRLDQ_128",

    VPSLLDQ_256       <- "VPSLLDQ_256",
    VPSRLDQ_256       <- "VPSRLDQ_256",

    VPUNPCKH_2u64     <- "VPUNPCKH_2u64",     
    VPUNPCKH_4u32     <- "VPUNPCKH_4u32",  
    VPUNPCKH_8u16     <- "VPUNPCKH_8u16",   
    VPUNPCKH_16u8     <- "VPUNPCKH_16u8",  

    VPUNPCKL_2u64     <- "VPUNPCKL_2u64",  
    VPUNPCKL_4u32     <- "VPUNPCKL_4u32",  
    VPUNPCKL_8u16     <- "VPUNPCKL_8u16",   
    VPUNPCKL_16u8     <- "VPUNPCKL_16u8",

    VPUNPCKH_4u64     <- "VPUNPCKH_4u64",     
    VPUNPCKH_8u32     <- "VPUNPCKH_8u32",  
    VPUNPCKH_16u16    <- "VPUNPCKH_16u16",   
    VPUNPCKH_32u8     <- "VPUNPCKH_32u8",  

    VPUNPCKL_4u64     <- "VPUNPCKL_4u64",  
    VPUNPCKL_8u32     <- "VPUNPCKL_8u32",  
    VPUNPCKL_16u16    <- "VPUNPCKL_16u16",   
    VPUNPCKL_32u8     <- "VPUNPCKL_32u8"

    from "../specs/avx2.spec".
