from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable

from autogen.codegen.compare_unprojected_generators import canonicalize_autogen
from autogen.codegen.projected_terms import (
    PlannedTerm,
    _parity_from_perm,
    _split_labels,
    canonicalize_projected_tensor,
    deserialize_coefficient,
    plan_structured_term,
)


StructuredTensor = tuple[str, str]


@dataclass(frozen=True)
class AntisymmetricOrbit:
    plan: PlannedTerm
    additions: tuple[tuple[float, tuple[int, ...]], ...]
    antisymmetrizer_coefficient: float


@dataclass(frozen=True)
class _GlobalCSEStep:
    step: Any
    slot: str
    tangent_slot: str | None
    arguments: tuple[str, ...]
    tangent_arguments: tuple[str | None, ...]
    first_plan: int
    last_plan: int


def _canonical_codegen_outputs(canonical_manifest: dict[str, Any]) -> dict[str, Any]:
    """Collect dummy-label and antisymmetry-equivalent Wick terms exactly."""

    canonical = canonicalize_autogen(canonical_manifest)
    external_labels = {"energy": "", "r1": "pq", "r2": "pqrs"}
    outputs: dict[str, Any] = {}
    for output_name, terms in canonical.items():
        serialized = []
        for tensors, coefficient in sorted(terms.items()):
            serialized.append(
                {
                    "output_labels": external_labels[output_name],
                    "tensors": [
                        {"name": name, "labels": labels}
                        for name, labels in tensors
                    ],
                    "coefficient": {
                        "numerator": int(coefficient.numerator),
                        "denominator": int(coefficient.denominator),
                    },
                }
            )
        outputs[output_name] = {"terms": serialized}
    return outputs


def _canonical_variant(
    term: dict[str, Any],
    output_tokens: tuple[str, ...],
) -> tuple[tuple[StructuredTensor, ...], tuple[str, ...], int]:
    variants: list[
        tuple[tuple[StructuredTensor, ...], tuple[str, ...], int]
    ] = []
    for permutation in permutations(output_tokens):
        output_map = dict(zip(output_tokens, permutation))
        tensor_sign = 1
        tensors: list[StructuredTensor] = []
        for tensor in term["tensors"]:
            labels = "".join(
                output_map.get(token, token)
                for token in _split_labels(tensor["labels"])
            )
            labels, sign = canonicalize_projected_tensor(tensor["name"], labels)
            tensor_sign *= sign
            tensors.append((tensor["name"], labels))
        variants.append((tuple(sorted(tensors)), permutation, tensor_sign))
    return min(variants, key=lambda item: (item[0], item[1]))


def build_antisymmetric_orbits(
    output_key: str,
    terms: Iterable[dict[str, Any]],
) -> tuple[AntisymmetricOrbit, ...]:
    terms = tuple(terms)
    if not terms:
        return ()
    output_labels = str(terms[0]["output_labels"])
    output_tokens = tuple(_split_labels(output_labels))
    if len(output_tokens) not in (2, 4):
        raise ValueError("antisymmetric output orbits require rank two or four")

    grouped: dict[
        tuple[StructuredTensor, ...],
        list[tuple[float, tuple[int, ...], float]],
    ] = defaultdict(list)
    for term in terms:
        if term["output_labels"] != output_labels:
            raise ValueError("all orbit terms must have the same output labels")
        tensors, permutation, tensor_sign = _canonical_variant(term, output_tokens)
        axes = tuple(output_tokens.index(token) for token in permutation)
        parity = _parity_from_perm(output_tokens, permutation)
        coefficient = deserialize_coefficient(term)
        grouped[tensors].append(
            (coefficient * tensor_sign, axes, coefficient * tensor_sign * parity)
        )

    orbits: list[AntisymmetricOrbit] = []
    for index, (tensors, entries) in enumerate(grouped.items()):
        invariant = [entry[2] for entry in entries]
        if max(invariant) - min(invariant) > 1.0e-10:
            raise ValueError(
                f"{output_key} orbit {index} has inconsistent antisymmetric coefficients"
            )
        plan = plan_structured_term(
            output_labels,
            list(tensors),
            name=f"{output_key}_orbit_{index}",
        )
        orbits.append(
            AntisymmetricOrbit(
                plan=plan,
                additions=tuple((coefficient, axes) for coefficient, axes, _ in entries),
                antisymmetrizer_coefficient=(
                    invariant[0] * len(entries) / (2.0 if len(output_tokens) == 2 else 24.0)
                ),
            )
        )
    return tuple(orbits)


def _build_global_cse(
    plans: Iterable[PlannedTerm],
    *,
    max_result_rank: int = 4,
) -> tuple[
    dict[Any, str],
    dict[Any, str],
    tuple[_GlobalCSEStep, ...],
    int,
]:
    """Select repeated low-rank contractions shared by all output orbits."""

    plans = tuple(plans)
    occurrences: list[tuple[Any, Any, int]] = []
    occurrence_plans: dict[Any, list[int]] = defaultdict(list)
    for plan_index, plan in enumerate(plans):
        runtime_keys: dict[str, Any] = {}
        for step in plan.steps:
            argument_keys = tuple(
                runtime_keys.get(reference, ("leaf", reference))
                for reference in step.arg_refs
            )
            runtime_key = ("einsum", step.expr, argument_keys)
            runtime_keys[step.name] = runtime_key
            occurrences.append((runtime_key, step, plan_index))
            occurrence_plans[runtime_key].append(plan_index)
    counts = Counter(runtime_key for runtime_key, _step, _plan in occurrences)
    representatives = {
        runtime_key: step for runtime_key, step, _plan in occurrences
    }
    selected = {
        signature
        for signature, count in counts.items()
        if count > 1
        and representatives[signature].result_rank <= max_result_rank
    }
    slots: dict[Any, str] = {}
    tangent_slots: dict[Any, str] = {}
    instructions: list[_GlobalCSEStep] = []
    tangent_inputs = {"t1": "dt1", "t2": "dt2"}
    for plan_index, plan in enumerate(plans):
        local_steps: dict[str, Any] = {}
        runtime_keys: dict[str, Any] = {}
        for step in plan.steps:
            local_steps[step.name] = step
            argument_keys = tuple(
                runtime_keys.get(reference, ("leaf", reference))
                for reference in step.arg_refs
            )
            runtime_key = ("einsum", step.expr, argument_keys)
            runtime_keys[step.name] = runtime_key
            if runtime_key not in selected or runtime_key in slots:
                continue
            arguments: list[str] = []
            tangent_arguments: list[str | None] = []
            for reference in step.arg_refs:
                child = local_steps.get(reference)
                if child is None:
                    arguments.append(reference)
                    tangent_arguments.append(tangent_inputs.get(reference))
                    continue
                child_key = runtime_keys[child.name]
                if child_key not in slots:
                    raise ValueError(
                        "global CSE selected a parent without its repeated child"
                    )
                arguments.append(slots[child_key])
                tangent_arguments.append(tangent_slots.get(child_key))
            index = len(instructions)
            slot = f"_cse{index}"
            tangent_slot = (
                f"_dcse{index}" if any(value is not None for value in tangent_arguments) else None
            )
            slots[runtime_key] = slot
            if tangent_slot is not None:
                tangent_slots[runtime_key] = tangent_slot
            instructions.append(
                _GlobalCSEStep(
                    step=step,
                    slot=slot,
                    tangent_slot=tangent_slot,
                    arguments=tuple(arguments),
                    tangent_arguments=tuple(tangent_arguments),
                    first_plan=min(occurrence_plans[runtime_key]),
                    last_plan=max(occurrence_plans[runtime_key]),
                )
            )
    saved_contractions = sum(counts[signature] - 1 for signature in selected)
    return slots, tangent_slots, tuple(instructions), saved_contractions


def _emit_global_cse(
    lines: list[str],
    instructions: Iterable[_GlobalCSEStep],
    *,
    with_jvp: bool,
    plan_index: int,
) -> None:
    for instruction in instructions:
        if instruction.first_plan != plan_index:
            continue
        optimize = (
            "_BINARY_EINSUM_PATH"
            if len(instruction.arguments) == 2
            else "True"
        )
        lines.append(
            f"    {instruction.slot} = "
            + _runtime_contraction(
                instruction.step.expr,
                instruction.arguments,
                optimize,
            )
        )
        if not with_jvp or instruction.tangent_slot is None:
            continue
        derivative_terms: list[str] = []
        for index, tangent in enumerate(instruction.tangent_arguments):
            if tangent is None:
                continue
            arguments = list(instruction.arguments)
            arguments[index] = tangent
            derivative_terms.append(
                _runtime_contraction(instruction.step.expr, arguments, optimize)
            )
        lines.append(
            f"    {instruction.tangent_slot} = " + " + ".join(derivative_terms)
        )


def _delete_expired_global_cse(
    lines: list[str],
    instructions: Iterable[_GlobalCSEStep],
    *,
    with_jvp: bool,
    plan_index: int,
) -> None:
    expired = [
        instruction
        for instruction in instructions
        if instruction.last_plan == plan_index
    ]
    slots = [instruction.slot for instruction in expired]
    if with_jvp:
        slots.extend(
            instruction.tangent_slot
            for instruction in expired
            if instruction.tangent_slot is not None
        )
    if slots:
        lines.append("    del " + ", ".join(slots))


def _emit_plan(
    lines: list[str],
    plan: PlannedTerm,
    shared_slots: dict[Any, str],
) -> tuple[str, tuple[str, ...]]:
    slots: dict[str, str] = {}
    runtime_keys: dict[str, Any] = {}
    owned_slots: list[str] = []
    for index, step in enumerate(plan.steps):
        argument_keys = tuple(
            runtime_keys.get(reference, ("leaf", reference))
            for reference in step.arg_refs
        )
        runtime_key = ("einsum", step.expr, argument_keys)
        runtime_keys[step.name] = runtime_key
        shared = shared_slots.get(runtime_key)
        if shared is not None:
            slots[step.name] = shared
            continue
        slot = f"_s{index}"
        slots[step.name] = slot
        owned_slots.append(slot)
        arguments = [slots.get(reference, reference) for reference in step.arg_refs]
        optimize = "_BINARY_EINSUM_PATH" if len(arguments) == 2 else "True"
        lines.append(
            f"    {slot} = " + _runtime_contraction(step.expr, arguments, optimize)
        )
    return slots.get(plan.root_ref, plan.root_ref), tuple(owned_slots)


def _emit_plan_with_jvp(
    lines: list[str],
    plan: PlannedTerm,
    shared_slots: dict[Any, str],
    shared_tangent_slots: dict[Any, str],
) -> tuple[str, str | None, tuple[str, ...], tuple[str, ...]]:
    slots: dict[str, str] = {}
    tangent_slots: dict[str, str] = {}
    runtime_keys: dict[str, Any] = {}
    owned_slots: list[str] = []
    owned_tangent_slots: list[str] = []
    tangent_inputs = {"t1": "dt1", "t2": "dt2"}
    for index, step in enumerate(plan.steps):
        argument_keys = tuple(
            runtime_keys.get(reference, ("leaf", reference))
            for reference in step.arg_refs
        )
        runtime_key = ("einsum", step.expr, argument_keys)
        runtime_keys[step.name] = runtime_key
        shared = shared_slots.get(runtime_key)
        if shared is not None:
            slots[step.name] = shared
            shared_tangent = shared_tangent_slots.get(runtime_key)
            if shared_tangent is not None:
                tangent_slots[step.name] = shared_tangent
            continue
        slot = f"_s{index}"
        tangent_slot = f"_ds{index}"
        slots[step.name] = slot
        owned_slots.append(slot)
        arguments = [slots.get(reference, reference) for reference in step.arg_refs]
        optimize = "_BINARY_EINSUM_PATH" if len(arguments) == 2 else "True"
        lines.append(
            f"    {slot} = " + _runtime_contraction(step.expr, arguments, optimize)
        )
        derivative_terms: list[str] = []
        for argument_index, reference in enumerate(step.arg_refs):
            tangent = tangent_slots.get(reference, tangent_inputs.get(reference))
            if tangent is None:
                continue
            derivative_arguments = list(arguments)
            derivative_arguments[argument_index] = tangent
            derivative_terms.append(
                _runtime_contraction(step.expr, derivative_arguments, optimize)
            )
        if derivative_terms:
            lines.append(f"    {tangent_slot} = " + " + ".join(derivative_terms))
            tangent_slots[step.name] = tangent_slot
            owned_tangent_slots.append(tangent_slot)
    root = slots.get(plan.root_ref, plan.root_ref)
    tangent_root = tangent_slots.get(plan.root_ref, tangent_inputs.get(plan.root_ref))
    return (
        root,
        tangent_root,
        tuple(owned_slots),
        tuple(owned_tangent_slots),
    )


def _runtime_contraction(
    expression: str,
    arguments: Iterable[str],
    optimize: str,
) -> str:
    arguments = tuple(arguments)
    # Packing is exact only when both operands are independently antisymmetric
    # in their row and contracted index pairs.  Some rank-four CSE tensors gain
    # antisymmetry only after their final output permutation and must remain on
    # the general einsum path.
    pair_packed_arguments = {
        ("_cse3", "h04"),
        ("_dcse3", "h04"),
        ("_s1", "_cse3"),
        ("_ds1", "_cse3"),
        ("_s1", "_dcse3"),
        ("h22", "_cse3"),
        ("h22", "_dcse3"),
    }
    if (
        len(arguments) == 2
        and expression == "abcd,efcd->abef"
        and arguments in pair_packed_arguments
    ):
        return f"_pair_pair_contract({arguments[0]}, {arguments[1]})"
    if (
        len(arguments) == 2
        and expression == "abcd,efab->efcd"
        and (arguments[1], arguments[0]) in pair_packed_arguments
    ):
        return f"_pair_pair_contract({arguments[1]}, {arguments[0]})"
    return (
        f"np.einsum({expression!r}, {', '.join(arguments)}, "
        f"optimize={optimize})"
    )


def write_antisymmetric_orbit_module(
    canonical_manifest: dict[str, Any],
    path: str | Path,
    *,
    tensor_names: Iterable[str],
) -> dict[str, int]:
    tensor_names = tuple(tensor_names)
    outputs = _canonical_codegen_outputs(canonical_manifest)
    energy_terms = tuple(outputs["energy"]["terms"])
    energy_plans = tuple(
        (
            plan_structured_term(
                "",
                [(tensor["name"], tensor["labels"]) for tensor in term["tensors"]],
                name=f"energy_{index}",
            ),
            deserialize_coefficient(term),
        )
        for index, term in enumerate(energy_terms)
    )
    r1_orbits = build_antisymmetric_orbits("r1", outputs["r1"]["terms"])
    r2_orbits = build_antisymmetric_orbits("r2", outputs["r2"]["terms"])
    all_plans = tuple(plan for plan, _coefficient in energy_plans) + tuple(
        orbit.plan for orbit in (*r1_orbits, *r2_orbits)
    )
    (
        shared_slots,
        shared_tangent_slots,
        cse_instructions,
        saved_contractions,
    ) = _build_global_cse(all_plans)
    (
        energy_shared_slots,
        _energy_shared_tangent_slots,
        energy_cse_instructions,
        _energy_saved_contractions,
    ) = _build_global_cse(plan for plan, _coefficient in energy_plans)
    maximum_live_cse = max(
        (
            sum(
                instruction.first_plan <= plan_index <= instruction.last_plan
                for instruction in cse_instructions
            )
            for plan_index in range(len(all_plans))
        ),
        default=0,
    )
    maximum_live_rank4_cse = max(
        (
            sum(
                instruction.step.result_rank == 4
                and instruction.first_plan <= plan_index <= instruction.last_plan
                for instruction in cse_instructions
            )
            for plan_index in range(len(all_plans))
        ),
        default=0,
    )
    contraction_count = sum(
        len(plan.steps) for plan, _coefficient in energy_plans
    ) + sum(
        len(orbit.plan.steps) for orbit in (*r1_orbits, *r2_orbits)
    )
    executed_contraction_count = contraction_count - saved_contractions

    signature = ", ".join(tensor_names)
    lines = [
        '"""Generated antisymmetric-orbit QPCCSD contractions; do not edit."""',
        "",
        "from __future__ import annotations",
        "",
        "import numpy as np",
        "",
        "_BINARY_EINSUM_PATH = ('einsum_path', (0, 1))",
        "_PAIR_INDEX_CACHE = {}",
        "CONTRACTION_BACKEND = 'pair-packed-preplanned-numpy-blas'",
        f"TENSOR_NAMES = {tensor_names!r}",
        "MAX_FORMAL_SCALING = 6",
        f"CONTRACTION_COUNT = {contraction_count}",
        f"EXECUTED_CONTRACTION_COUNT = {executed_contraction_count}",
        f"GLOBAL_CSE_INTERMEDIATE_COUNT = {len(cse_instructions)}",
        f"MAX_LIVE_GLOBAL_CSE = {maximum_live_cse}",
        f"MAX_LIVE_RANK4_CSE = {maximum_live_rank4_cse}",
        f"PAIR_ORBIT_COUNT = {len(r1_orbits)}",
        f"QUADRUPLE_ORBIT_COUNT = {len(r2_orbits)}",
        "",
        "def _antisymmetrize_pair(tensor):",
        "    return tensor - tensor.T",
        "",
        "def _antisymmetrize_rank4(tensor):",
        "    pair_antisymmetric = tensor - tensor.swapaxes(0, 1)",
        "    pair_antisymmetric = pair_antisymmetric - pair_antisymmetric.swapaxes(2, 3)",
        "    return (",
        "        pair_antisymmetric",
        "        - pair_antisymmetric.transpose((0, 2, 1, 3))",
        "        + pair_antisymmetric.transpose((0, 2, 3, 1))",
        "        + pair_antisymmetric.transpose((2, 0, 1, 3))",
        "        - pair_antisymmetric.transpose((2, 0, 3, 1))",
        "        + pair_antisymmetric.transpose((2, 3, 0, 1))",
        "    )",
        "",
        "def _pair_pair_contract(left, right):",
        "    dimension = left.shape[0]",
        "    indices = _PAIR_INDEX_CACHE.get(dimension)",
        "    if indices is None:",
        "        indices = np.triu_indices(dimension, 1)",
        "        _PAIR_INDEX_CACHE[dimension] = indices",
        "    first, second = indices",
        "    rows = first[:, None]",
        "    columns = second[:, None]",
        "    packed_left = left[rows, columns, first[None, :], second[None, :]]",
        "    packed_right = right[rows, columns, first[None, :], second[None, :]]",
        "    packed = 2.0 * (packed_left @ packed_right.T)",
        "    result = np.zeros((dimension,) * 4, dtype=np.result_type(left, right))",
        "    result[rows, columns, first[None, :], second[None, :]] = packed",
        "    result[columns, rows, first[None, :], second[None, :]] = -packed",
        "    result[rows, columns, second[None, :], first[None, :]] = -packed",
        "    result[columns, rows, second[None, :], first[None, :]] = packed",
        "    return result",
        "",
        f"def compute_outputs({signature}):",
    ]
    for name in tensor_names:
        lines.append(f"    {name} = np.asarray({name})")
    lines.extend(
        [
            f"    dtype = np.result_type({signature}, np.complex128)",
            "    energy = np.array(0.0 + 0.0j, dtype=dtype)",
            "    r1 = np.zeros((t1.shape[0], t1.shape[0]), dtype=dtype)",
            "    r2 = np.zeros((t1.shape[0],) * 4, dtype=dtype)",
        ]
    )

    plan_index = 0
    for plan, coefficient in energy_plans:
        _emit_global_cse(
            lines,
            cse_instructions,
            with_jvp=False,
            plan_index=plan_index,
        )
        root, slots = _emit_plan(lines, plan, shared_slots)
        lines.append(f"    energy += ({coefficient!r}) * {root}")
        if slots:
            lines.append("    del " + ", ".join(slots))
        _delete_expired_global_cse(
            lines,
            cse_instructions,
            with_jvp=False,
            plan_index=plan_index,
        )
        plan_index += 1

    for output_name, orbits in (("r1", r1_orbits), ("r2", r2_orbits)):
        antisymmetrizer = (
            "_antisymmetrize_pair" if output_name == "r1" else "_antisymmetrize_rank4"
        )
        rank = 2 if output_name == "r1" else 4
        identity_axes = tuple(range(rank))
        for orbit in orbits:
            _emit_global_cse(
                lines,
                cse_instructions,
                with_jvp=False,
                plan_index=plan_index,
            )
            root, slots = _emit_plan(lines, orbit.plan, shared_slots)
            use_factored_antisymmetrizer = output_name == "r2" and len(orbit.additions) > 6
            if use_factored_antisymmetrizer:
                lines.append(
                    f"    {output_name} += ({orbit.antisymmetrizer_coefficient!r}) "
                    f"* {antisymmetrizer}({root})"
                )
            else:
                for coefficient, axes in orbit.additions:
                    value = (
                        root
                        if axes == identity_axes
                        else f"{root}.transpose({axes!r})"
                    )
                    lines.append(
                        f"    {output_name} += ({coefficient!r}) * {value}"
                    )
            if slots:
                lines.append("    del " + ", ".join(slots))
            _delete_expired_global_cse(
                lines,
                cse_instructions,
                with_jvp=False,
                plan_index=plan_index,
            )
            plan_index += 1

    lines.extend(
        [
            "    return {",
            "        'energy': np.real_if_close(energy),",
            "        'r1': np.real_if_close(r1),",
            "        'r2': np.real_if_close(r2),",
            "    }",
            "",
            f"def compute_outputs_and_jvp(t1, t2, dt1, dt2, {', '.join(tensor_names[2:])}):",
            "    t1 = np.asarray(t1)",
            "    t2 = np.asarray(t2)",
            "    dt1 = np.asarray(dt1)",
            "    dt2 = np.asarray(dt2)",
        ]
    )
    for name in tensor_names[2:]:
        lines.append(f"    {name} = np.asarray({name})")
    lines.extend(
        [
            f"    dtype = np.result_type({signature}, dt1, dt2, np.complex128)",
            "    energy = np.array(0.0 + 0.0j, dtype=dtype)",
            "    r1 = np.zeros((t1.shape[0], t1.shape[0]), dtype=dtype)",
            "    r2 = np.zeros((t1.shape[0],) * 4, dtype=dtype)",
            "    energy_jvp = np.array(0.0 + 0.0j, dtype=dtype)",
            "    r1_jvp = np.zeros((t1.shape[0], t1.shape[0]), dtype=dtype)",
            "    r2_jvp = np.zeros((t1.shape[0],) * 4, dtype=dtype)",
        ]
    )

    plan_index = 0
    for plan, coefficient in energy_plans:
        _emit_global_cse(
            lines,
            cse_instructions,
            with_jvp=True,
            plan_index=plan_index,
        )
        root, tangent_root, slots, tangent_slots = _emit_plan_with_jvp(
            lines,
            plan,
            shared_slots,
            shared_tangent_slots,
        )
        lines.append(f"    energy += ({coefficient!r}) * {root}")
        if tangent_root is not None:
            lines.append(f"    energy_jvp += ({coefficient!r}) * {tangent_root}")
        if slots or tangent_slots:
            lines.append("    del " + ", ".join((*slots, *tangent_slots)))
        _delete_expired_global_cse(
            lines,
            cse_instructions,
            with_jvp=True,
            plan_index=plan_index,
        )
        plan_index += 1

    for output_name, orbits in (("r1", r1_orbits), ("r2", r2_orbits)):
        antisymmetrizer = (
            "_antisymmetrize_pair" if output_name == "r1" else "_antisymmetrize_rank4"
        )
        rank = 2 if output_name == "r1" else 4
        identity_axes = tuple(range(rank))
        for orbit in orbits:
            _emit_global_cse(
                lines,
                cse_instructions,
                with_jvp=True,
                plan_index=plan_index,
            )
            root, tangent_root, slots, tangent_slots = _emit_plan_with_jvp(
                lines,
                orbit.plan,
                shared_slots,
                shared_tangent_slots,
            )
            use_factored_antisymmetrizer = output_name == "r2" and len(orbit.additions) > 6
            if use_factored_antisymmetrizer:
                lines.append(
                    f"    {output_name} += "
                    f"({orbit.antisymmetrizer_coefficient!r}) "
                    f"* {antisymmetrizer}({root})"
                )
                if tangent_root is not None:
                    lines.append(
                        f"    {output_name}_jvp += "
                        f"({orbit.antisymmetrizer_coefficient!r}) "
                        f"* {antisymmetrizer}({tangent_root})"
                    )
            else:
                for coefficient, axes in orbit.additions:
                    value = (
                        root
                        if axes == identity_axes
                        else f"{root}.transpose({axes!r})"
                    )
                    lines.append(
                        f"    {output_name} += ({coefficient!r}) * {value}"
                    )
                    if tangent_root is not None:
                        tangent_value = (
                            tangent_root
                            if axes == identity_axes
                            else f"{tangent_root}.transpose({axes!r})"
                        )
                        lines.append(
                            f"    {output_name}_jvp += ({coefficient!r}) "
                            f"* {tangent_value}"
                        )
            if slots or tangent_slots:
                lines.append("    del " + ", ".join((*slots, *tangent_slots)))
            _delete_expired_global_cse(
                lines,
                cse_instructions,
                with_jvp=True,
                plan_index=plan_index,
            )
            plan_index += 1

    lines.extend(
        [
            "    return {",
            "        'energy': np.real_if_close(energy),",
            "        'r1': np.real_if_close(r1),",
            "        'r2': np.real_if_close(r2),",
            "        'energy_jvp': np.real_if_close(energy_jvp),",
            "        'r1_jvp': np.real_if_close(r1_jvp),",
            "        'r2_jvp': np.real_if_close(r2_jvp),",
            "    }",
            "",
            f"def compute_jvp(t1, t2, dt1, dt2, {', '.join(tensor_names[2:])}):",
            f"    outputs = compute_outputs_and_jvp(t1, t2, dt1, dt2, {', '.join(tensor_names[2:])})",
            "    return {",
            "        'energy': outputs['energy_jvp'],",
            "        'r1': outputs['r1_jvp'],",
            "        'r2': outputs['r2_jvp'],",
            "    }",
            "",
            f"def compute_energy({signature}):",
        ]
    )
    for name in tensor_names:
        lines.append(f"    {name} = np.asarray({name})")
    lines.extend(
        [
            f"    dtype = np.result_type({signature}, np.complex128)",
            "    energy = np.array(0.0 + 0.0j, dtype=dtype)",
        ]
    )
    for plan_index, (plan, coefficient) in enumerate(energy_plans):
        _emit_global_cse(
            lines,
            energy_cse_instructions,
            with_jvp=False,
            plan_index=plan_index,
        )
        root, slots = _emit_plan(lines, plan, energy_shared_slots)
        lines.append(f"    energy += ({coefficient!r}) * {root}")
        if slots:
            lines.append("    del " + ", ".join(slots))
        _delete_expired_global_cse(
            lines,
            energy_cse_instructions,
            with_jvp=False,
            plan_index=plan_index,
        )
    lines.extend(
        [
            "    return np.real_if_close(energy)",
            "",
            f"def compute_r1({signature}):",
            f"    return compute_outputs({signature})['r1']",
            "",
            f"def compute_r2({signature}):",
            f"    return compute_outputs({signature})['r2']",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines))
    return {
        "energy_terms": len(energy_terms),
        "pair_orbits": len(r1_orbits),
        "quadruple_orbits": len(r2_orbits),
        "contraction_count": contraction_count,
        "executed_contraction_count": executed_contraction_count,
        "global_cse_intermediate_count": len(cse_instructions),
        "maximum_live_cse": maximum_live_cse,
        "maximum_live_rank4_cse": maximum_live_rank4_cse,
    }
