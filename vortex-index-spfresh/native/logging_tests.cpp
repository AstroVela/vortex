// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <iostream>

#include "inc/Core/Common.h"
#include "inc/Core/Common/InstructionUtils.h"

int main() {
    SPTAG::COMMON::InstructionSet::PrintInstructionSet();
    SPTAG::g_pLogger->Logging("SPTAG",
                              SPTAG::Helper::LogLevel::LL_Warning,
                              __FILE__,
                              __LINE__,
                              __func__,
                              "host-sdk-log\n");
    std::cout << "{\"ready\":true}\n";
}
