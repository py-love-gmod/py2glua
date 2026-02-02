from .comp_const import FoldCompileTimeBoolConstsPass
from .const_folding import ConstFoldingPass
from .dce import DcePass
from .gmod_api import (
    CollectGmodApiDeclsPass,
    FinalizeGmodApiRegistryPass,
    RewriteGmodApiCallsPass,
)
from .nil_fold import NilFoldPass
from .strip_asign import StripPythonOnlyNodesPass
from .strip_cd import StripCompilerDirectiveDefPass
from .strip_no_compile_gmod_api import StripNoCompileAndGmodApiDefsPass
