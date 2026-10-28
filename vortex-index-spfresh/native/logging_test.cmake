# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
execute_process(COMMAND "${PROGRAM}" RESULT_VARIABLE status
                OUTPUT_VARIABLE output ERROR_VARIABLE diagnostics)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "Native startup probe failed: ${status}: ${diagnostics}")
endif()
if(NOT output STREQUAL "{\"ready\":true}\n")
    message(FATAL_ERROR "Native SDK diagnostics polluted host stdout: ${output}")
endif()
if(NOT diagnostics MATCHES "host-sdk-log")
    message(FATAL_ERROR "Native SDK logger did not emit diagnostics to stderr")
endif()
