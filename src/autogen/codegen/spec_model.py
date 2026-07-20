"""Typed theory spec loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import os

from .compat import load_spec_namespace


TENSOR_MAP_DEFAULT = {
    "V2": "g",
    "F1": "f",
    "T1": "t1",
    "T2": "t2",
    "R1": "r1",
    "R2": "r2",
    "D1": "d1",
    "D2": "d2",
    "X1": "x1",
    "X2": "x2",
}

SUPPORTED_CAPABILITIES = {
    "standard_bch",
    "eom",
    "qp",
    "projected_qp",
    "spin_orbital",
    "spin_adapted",
}


@dataclass(frozen=True)
class ParsedTerm:
    output_key: str
    factor: float
    ops: tuple[str, ...]


@dataclass(frozen=True)
class MethodSpec:
    theory_name: str
    spec_path: str | None
    terms: tuple[ParsedTerm, ...]
    outputs: tuple[tuple[str, str], ...]
    tensor_map: tuple[tuple[str, str], ...]
    view_tensors: tuple[str, ...]
    output_dir: str | None
    tasks: tuple[str, ...]
    pyscf_mol: dict[str, Any] | None
    spin_orbital: bool
    spin_adapted: bool
    bogoliubov_qp: bool
    bch: bool
    eom_bch: bool
    backend_capabilities: tuple[str, ...]
    runtime_options: tuple[tuple[str, Any], ...]
    oracle_available: bool
    oracle_factory: str | None

    @property
    def output_names(self) -> dict[str, str]:
        return dict(self.outputs)

    @property
    def tensor_names(self) -> dict[str, str]:
        return dict(self.tensor_map)

    @property
    def runtime_options_map(self) -> dict[str, Any]:
        return dict(self.runtime_options)

    def legacy_parse_result(self):
        return (
            [
                {
                    "output_key": term.output_key,
                    "fac": term.factor,
                    "ops": list(term.ops),
                }
                for term in self.terms
            ],
            self.output_names,
            self.tensor_names,
            list(self.view_tensors),
            self.output_dir,
            list(self.tasks),
            self.pyscf_mol,
            self.spin_orbital,
            self.spin_adapted,
        )


def _infer_output_key(ops: list[str]) -> str:
    projector = [op for op in ops if op.startswith("X") and op[1:].isdigit()]
    if not projector:
        return "scalar"
    if len(projector) != 1:
        raise ValueError(f"Unable to infer output from ops {ops}: multiple X projectors present.")
    return projector[0]


def default_output_name(output_key: str) -> str:
    if output_key.startswith("X") and output_key[1:].isdigit():
        return f"r{output_key[1:]}"
    if output_key == "scalar":
        return "scalar"
    return output_key.lower()


def resolve_output_name(output_key: str, output_names: Mapping[str, str]) -> str:
    name = output_names.get(output_key)
    if name is None:
        name = default_output_name(output_key)
    if not str(name).isidentifier():
        raise ValueError(f"Invalid output name '{name}' for '{output_key}'.")
    return str(name)


def _normalize_output_names(spec: Mapping[str, Any]) -> dict[str, str]:
    output_names = spec.get("OUTPUTS", {})
    if output_names is None:
        output_names = {}
    if not isinstance(output_names, dict):
        raise ValueError("OUTPUTS must be a dict.")
    normalized = {str(k): str(v) for k, v in output_names.items()}
    for key in normalized:
        resolve_output_name(key, normalized)
    return normalized


def _normalize_tensor_map(spec: Mapping[str, Any]) -> dict[str, str]:
    tensor_map = dict(TENSOR_MAP_DEFAULT)
    overrides = spec.get("TENSOR_MAP")
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError("TENSOR_MAP must be a dict.")
        tensor_map.update({str(k): str(v) for k, v in overrides.items()})
    return tensor_map


def _normalize_view_tensors(spec: Mapping[str, Any]) -> tuple[str, ...]:
    view_tensors = spec.get("VIEW_TENSORS", ("g", "f"))
    if isinstance(view_tensors, str):
        view_tensors = [view_tensors]
    if not isinstance(view_tensors, (list, tuple)):
        raise ValueError("VIEW_TENSORS must be a list or tuple.")
    return tuple(str(name) for name in view_tensors)


def _normalize_tasks(spec: Mapping[str, Any]) -> tuple[str, ...]:
    tasks = spec.get("TASKS", [])
    if tasks is None:
        tasks = []
    if isinstance(tasks, str):
        tasks = [tasks]
    if not isinstance(tasks, (list, tuple)):
        raise ValueError("TASKS must be a list or tuple.")
    return tuple(str(task) for task in tasks)


def _normalize_pyscf_mol(spec: Mapping[str, Any]) -> dict[str, Any] | None:
    pyscf_mol = spec.get("PYSCF_MOL")
    if pyscf_mol is None:
        return None
    if not isinstance(pyscf_mol, dict):
        raise ValueError("PYSCF_MOL must be a dict.")
    return {str(k): v for k, v in pyscf_mol.items()}


def _parse_terms(spec: Mapping[str, Any]) -> tuple[ParsedTerm, ...]:
    raw_terms = spec.get("TERMS")
    if raw_terms is None:
        if spec.get("EOM_BCH") or spec.get("BCH"):
            raw_terms = []
        else:
            raise ValueError("Spec file must define TERMS.")
    if not isinstance(raw_terms, (list, tuple)):
        raise ValueError("TERMS must be a list or tuple.")

    parsed: list[ParsedTerm] = []
    for item in raw_terms:
        output_key = None
        if isinstance(item, dict):
            ops = item.get("ops")
            factor = item.get("fac", 1.0)
            output_key = item.get("output")
        elif isinstance(item, (list, tuple)):
            if len(item) == 2:
                factor, ops = item
            elif len(item) == 3:
                output_key, factor, ops = item
            else:
                raise ValueError("Term tuples must have 2 or 3 entries.")
        else:
            raise ValueError("Each term must be a dict or tuple.")

        if ops is None:
            raise ValueError("Each term must define ops.")
        if not isinstance(ops, (list, tuple)):
            raise ValueError("ops must be a list or tuple.")

        ops_list = [str(op) for op in ops]
        key = _infer_output_key(ops_list) if output_key is None else str(output_key)
        parsed.append(ParsedTerm(output_key=key, factor=float(factor), ops=tuple(ops_list)))
    return tuple(parsed)


def _infer_capabilities(spec: Mapping[str, Any], *, spin_orbital: bool, spin_adapted: bool) -> tuple[str, ...]:
    caps = set()
    if spec.get("BCH"):
        caps.add("standard_bch")
    if spec.get("EOM_BCH"):
        caps.add("eom")
    if spec.get("BOGOLIUBOV_QP"):
        caps.add("qp")
    if spec.get("PROJECTED_QP"):
        caps.add("projected_qp")
    if spin_orbital:
        caps.add("spin_orbital")
    if spin_adapted:
        caps.add("spin_adapted")
    extra = spec.get("BACKEND_CAPABILITIES", ())
    if isinstance(extra, str):
        extra = [extra]
    if not isinstance(extra, (list, tuple, set)):
        raise ValueError("BACKEND_CAPABILITIES must be a sequence when provided.")
    caps.update(str(item) for item in extra)
    unknown = caps.difference(SUPPORTED_CAPABILITIES)
    if unknown:
        raise ValueError(f"Unknown backend capabilities: {sorted(unknown)}")
    return tuple(sorted(caps))


def namespace_to_method_spec(spec: Mapping[str, Any], *, spec_path: str | Path | None = None) -> MethodSpec:
    spin_orbital = bool(spec.get("SPIN_ORBITAL", False))
    spin_adapted = bool(spec.get("SPIN_ADAPTED", False))
    output_dir = spec.get("OUTPUT_DIR")
    if output_dir is not None:
        output_dir = str(output_dir)

    theory_name = str(
        spec.get("THEORY_NAME")
        or spec.get("METHOD_NAME")
        or (Path(output_dir).name if output_dir else "")
        or (Path(spec_path).stem if spec_path is not None else "unnamed_theory")
    )
    runtime_options = spec.get("RUNTIME_OPTIONS", {})
    if runtime_options is None:
        runtime_options = {}
    if not isinstance(runtime_options, dict):
        raise ValueError("RUNTIME_OPTIONS must be a dict when provided.")

    oracle_factory = spec.get("ORACLE_FACTORY")
    if oracle_factory is not None:
        oracle_factory = str(oracle_factory)

    return MethodSpec(
        theory_name=theory_name,
        spec_path=None if spec_path is None else str(spec_path),
        terms=_parse_terms(spec),
        outputs=tuple(sorted(_normalize_output_names(spec).items())),
        tensor_map=tuple(sorted(_normalize_tensor_map(spec).items())),
        view_tensors=_normalize_view_tensors(spec),
        output_dir=output_dir,
        tasks=_normalize_tasks(spec),
        pyscf_mol=_normalize_pyscf_mol(spec),
        spin_orbital=spin_orbital,
        spin_adapted=spin_adapted,
        bogoliubov_qp=bool(spec.get("BOGOLIUBOV_QP", False)),
        bch=bool(spec.get("BCH", False)),
        eom_bch=bool(spec.get("EOM_BCH", False)),
        backend_capabilities=_infer_capabilities(spec, spin_orbital=spin_orbital, spin_adapted=spin_adapted),
        runtime_options=tuple(sorted(runtime_options.items())),
        oracle_available=bool(spec.get("ORACLE_FACTORY") or spec.get("HAS_ORACLE", False)),
        oracle_factory=oracle_factory,
    )


def load_method_spec(spec_path) -> MethodSpec:
    namespace = load_spec_namespace(spec_path)
    spec = namespace_to_method_spec(namespace, spec_path=spec_path)
    if spec.bogoliubov_qp:
        os.environ["AUTOGEN_BOGOLIUBOV_QP"] = "1"
    else:
        os.environ.pop("AUTOGEN_BOGOLIUBOV_QP", None)
    return spec


def parse_legacy_spec_terms(spec: Mapping[str, Any]):
    return namespace_to_method_spec(spec).legacy_parse_result()
