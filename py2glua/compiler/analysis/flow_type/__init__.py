from plg_reader import IRFile

from ..symtable import SymbolTable
from .inferrer import TypeInferrer
from .result import InferenceResult


def infer_types(
    ir_file: IRFile,
    symbol_table: SymbolTable,
    external_tables: dict[str, SymbolTable] | None = None,
) -> InferenceResult:
    return TypeInferrer(symbol_table, external_tables).infer_file(ir_file)
