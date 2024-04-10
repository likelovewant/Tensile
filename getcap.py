#!/usr/bin/env python3

import os 
import subprocess
import re
import sys

def isExe(filePath):
    return os.path.isfile(filePath) and os.access(filePath, os.X_OK)


def gfxArch(name):
    match = re.search(r"gfx([0-9a-fA-F]{3,})", name)
    if not match:
        return None

    ipart = match.group(1)

    step = int(ipart[-1], 16)
    ipart = ipart[:-1]

    minor = int(ipart[-1])
    ipart = ipart[:-1]

    major = int(ipart)

    rv = (major, minor, step)

    return rv


def gfxName(arch):
    # convert last digit to hex because reasons
    name = str(arch[0]) + str(arch[1]) + ("%x" % arch[2])
    return "gfx" + "".join(map(str, name))


def locateExe(defaultPath, exeName):  # /opt/rocm/bin, hip-clang
    # look in defaultPath first
    exePath = os.path.join(defaultPath, exeName)
    if isExe(exePath):
        return exePath
    # look in PATH second
    for path in os.environ["PATH"].split(os.pathsep):
        exePath = os.path.join(path, exeName)
        if isExe(exePath):
            return exePath
    return None


ASSEMBLER_PATH = locateExe(os.path.join("/opt/rocm", "llvm/bin"), "clang++")

def tryAssembler(isaVersion, asmString, debug=False, *options):
    """
    Try to assemble the asmString for the specified target processor
    Success is defined as assembler returning no error code or stderr/stdout
    """
    options = list(options)
    debug = False

    if isaVersion[0] >= 10:
        options += ["-mwavefrontsize64"]

    assembler = ASSEMBLER_PATH
    if assembler is None:
        raise ValueError(
            "No assembler available; set TENSILE_ROCM_ASSEMBLER_PATH to point to ROCm Clang."
        )
    args = [
        assembler,
        "-x",
        "assembler",
        "-target",
        "amdgcn-amdhsa",
        "-mcpu=" + gfxName(isaVersion),
        *options,
        "-",
    ]

    result = subprocess.run(
        args, input=asmString.encode(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    output = result.stdout.decode()

    if debug:
        print("isaVersion: ", isaVersion)
        print("asm_cmd:", " ".join(args))
        print("asmString: ", asmString)
        print("output: ", output)
        print("return code: ", result.returncode)

    if output != "" or result.returncode != 0:
        return False
    return True


def GetAsmCaps(isaVersion):
    """Determine assembler capabilities by testing short instructions sequences"""

    derivedAsmCaps = {}
    derivedAsmCaps["SupportedISA"] = tryAssembler(isaVersion, "")
    derivedAsmCaps["HasExplicitCO"] = tryAssembler(
        isaVersion, "v_add_co_u32 v0,vcc,v0,1"
    )
    derivedAsmCaps["HasExplicitNC"] = tryAssembler(
        isaVersion, "v_add_nc_u32 v0,v0,1"
    )

    # Syntax of DirectToLds loads has changed: destination vgpr should be omitted
    # Old syntax should be removed in a future update as it is no longer supported
    derivedAsmCaps["HasDirectToLdsDest"] = tryAssembler(
        isaVersion, "buffer_load_dword v40, v36, s[24:27], s28 offen offset:0 lds"
    ) or tryAssembler(
        isaVersion, "buffer_load_b32 v40, v36, s[24:27], s28 offen offset:0 lds"
    )
    derivedAsmCaps["HasDirectToLdsNoDest"] = tryAssembler(
        isaVersion, "buffer_load_dword v36, s[24:27], s28 offen offset:0 lds"
    ) or tryAssembler(
        isaVersion, "buffer_load_b32 v36, s[24:27], s28 offen offset:0 lds"
    )

    derivedAsmCaps["HasAddLshl"] = tryAssembler(
        isaVersion, "v_add_lshl_u32 v47, v36, v34, 0x2"
    )
    derivedAsmCaps["HasLshlOr"] = tryAssembler(
        isaVersion, "v_lshl_or_b32 v47, v36, 0x2, v34"
    )
    derivedAsmCaps["HasSMulHi"] = tryAssembler(
        isaVersion, "s_mul_hi_u32 s47, s36, s34"
    )

    derivedAsmCaps["HasWMMA"] = tryAssembler(
        isaVersion, "v_wmma_f32_16x16x16_f16 v[0:7], v[8:15], v[16:23], v[0:7]"
    )
    derivedAsmCaps["HasMFMA"] = tryAssembler(
        isaVersion, "v_mfma_f32_32x32x2bf16 a[0:31], v32, v33, a[0:31]"
    ) or tryAssembler(
        isaVersion, "v_mfma_f32_32x32x1_2b_f32 a[0:31], v0, v1, a[0:31]"
    )
    derivedAsmCaps["HasMFMA_vgpr"] = tryAssembler(
        isaVersion, "v_mfma_f32_32x32x2bf16 v[0:31], v32, v33, v[0:31]"
    ) or tryAssembler(
        isaVersion, "v_mfma_f32_32x32x1_2b_f32 v[0:31], v0, v1, v[0:31]"
    )
    derivedAsmCaps["HasMFMA_f64"] = tryAssembler(
        isaVersion, "v_mfma_f64_16x16x4f64 v[0:7], v[32:33], v[36:37], v[0:7]"
    ) or tryAssembler(
        isaVersion, "v_mfma_f64_16x16x4_f64 v[0:7], v[32:33], v[36:37], v[0:7]"
    )
    derivedAsmCaps["HasMFMA_bf16_original"] = tryAssembler(
        isaVersion, "v_mfma_f32_32x32x2bf16 a[0:31], v32, v33, a[0:31]"
    )
    derivedAsmCaps["HasMFMA_bf16_1k"] = tryAssembler(
        isaVersion, "v_mfma_f32_32x32x4bf16_1k a[0:31], v[32:33], v[36:37], a[0:31]"
    )
    derivedAsmCaps["HasMFMA_xf32"] = tryAssembler(
        isaVersion, "v_mfma_f32_32x32x4_xf32 a[0:15], v[32:33], v[36:37], a[0:15]"
    )
    derivedAsmCaps["HasMFMA_f8"] = tryAssembler(
        isaVersion, "v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], a[0:3]"
    )
    derivedAsmCaps["HasMFMA_b8"] = tryAssembler(
        isaVersion, "v_mfma_f32_16x16x32_bf8_bf8 a[0:3], v[2:3], v[4:5], a[0:3]"
    )
    derivedAsmCaps["HasMFMA_i8_908"] = tryAssembler(
        isaVersion, "v_mfma_i32_32x32x8i8 a[0:15], v2, v3, a[0:15]"
    )
    derivedAsmCaps["HasMFMA_i8_940"] = tryAssembler(
        isaVersion, "v_mfma_i32_32x32x16_i8 a[0:15], v[2:3], v[4:5], a[0:15]"
    )

    derivedAsmCaps["v_mac_f16"] = tryAssembler(
        isaVersion, "v_mac_f16 v47, v36, v34"
    )

    derivedAsmCaps["v_fma_f16"] = tryAssembler(
        isaVersion, "v_fma_f16 v47, v36, v34, v47, op_sel:[0,0,0,0]"
    )
    derivedAsmCaps["v_fmac_f16"] = tryAssembler(
        isaVersion, "v_fma_f16 v47, v36, v34"
    )

    derivedAsmCaps["v_pk_fma_f16"] = tryAssembler(
        isaVersion, "v_pk_fma_f16 v47, v36, v34, v47, op_sel:[0,0,0]"
    )
    derivedAsmCaps["v_pk_fmac_f16"] = tryAssembler(
        isaVersion, "v_pk_fma_f16 v47, v36, v34"
    )

    derivedAsmCaps["v_mad_mix_f32"] = tryAssembler(
        isaVersion,
        "v_mad_mix_f32 v47, v36, v34, v47, op_sel:[0,0,0] op_sel_hi:[1,1,0]",
    )
    derivedAsmCaps["v_fma_mix_f32"] = tryAssembler(
        isaVersion,
        "v_fma_mix_f32 v47, v36, v34, v47, op_sel:[0,0,0] op_sel_hi:[1,1,0]",
    )

    derivedAsmCaps["v_dot2_f32_f16"] = tryAssembler(
        isaVersion, "v_dot2_f32_f16 v20, v36, v34, v20"
    )
    derivedAsmCaps["v_dot2c_f32_f16"] = tryAssembler(
        isaVersion, "v_dot2c_f32_f16 v47, v36, v34"
    ) or tryAssembler(isaVersion, "v_dot2acc_f32_f16 v47, v36, v34")

    derivedAsmCaps["v_dot4_i32_i8"] = tryAssembler(
        isaVersion, "v_dot4_i32_i8 v47, v36, v34"
    )
    derivedAsmCaps["v_dot4c_i32_i8"] = tryAssembler(
        isaVersion, "v_dot4c_i32_i8 v47, v36, v34"
    )
    derivedAsmCaps["VOP3v_dot4_i32_i8"] = tryAssembler(
        isaVersion, "v_dot4_i32_i8 v47, v36, v34, v47"
    )

    derivedAsmCaps["v_mac_f32"] = tryAssembler(
        isaVersion, "v_mac_f32 v20, v21, v22"
    )
    derivedAsmCaps["v_fma_f32"] = tryAssembler(
        isaVersion, "v_fma_f32 v20, v21, v22, v23"
    )
    derivedAsmCaps["v_fmac_f32"] = tryAssembler(
        isaVersion, "v_fmac_f32 v20, v21, v22"
    )

    derivedAsmCaps["v_fma_f64"] = tryAssembler(
        isaVersion, "v_fma_f64 v[20:21], v[22:23], v[24:25], v[20:21]"
    )

    derivedAsmCaps["HasAtomicAdd"] = tryAssembler(
        isaVersion, "buffer_atomic_add_f32 v0, v1, s[0:3], 0 offen offset:0"
    )
    derivedAsmCaps["HasGLCModifier"] = tryAssembler(
        isaVersion,
        "buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, glc",
    )

    if tryAssembler(isaVersion, "s_waitcnt vmcnt(63)"):
        derivedAsmCaps["MaxVmcnt"] = 63
    elif tryAssembler(isaVersion, "s_waitcnt vmcnt(15)"):
        derivedAsmCaps["MaxVmcnt"] = 15
    else:
        derivedAsmCaps["MaxVmcnt"] = 0

    # TODO- Need to query the max cap, just like vmcnt as well?
    derivedAsmCaps["MaxLgkmcnt"] = 15

    derivedAsmCaps["SupportedSource"] = True

    return derivedAsmCaps

if __name__ == "__main__":
    if len(sys.argv) > 1:
        gcn_arch = sys.argv[1]
    else:
        gcn_arch = "gfx900"

    print(GetAsmCaps(gfxArch(gcn_arch)))

