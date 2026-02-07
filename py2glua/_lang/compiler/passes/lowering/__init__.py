from .comp_const import FoldCompileTimeBoolConstsPass
from .const_folding import ConstFoldingPass
from .dce import DcePass
from .debug_compile_only import (
    CollectDebugCompileOnlyDeclsPass,
    RewriteAndStripDebugCompileOnlyPass,
)
from .enum_fold import CollectGmodSpecialEnumDeclsPass, FoldGmodSpecialEnumUsesPass
from .gmod_api import (
    CollectGmodApiDeclsPass,
    FinalizeGmodApiRegistryPass,
    RewriteGmodApiCallsPass,
)
from .lazy_compile import (
    CountSymlinkUsesPass,
    StripLazyCompileUnusedDefsPass,
)
from .nil_fold import NilFoldPass
from .raw import RewriteRawCallsPass
from .strip_asign import StripPythonOnlyNodesPass
from .strip_cd import StripCompilerDirectiveDefPass
from .strip_comments_imports import StripCommentsImportsPass
from .strip_enums import StripEnumsAndGmodSpecialEnumDefsPass
from .strip_no_compile_gmod_api import StripNoCompileAndGmodApiDefsPass
from .with_condition import (
    CollectWithConditionClassesPass,
    RewriteWithConditionBlocksPass,
)
