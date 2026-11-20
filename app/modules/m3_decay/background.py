from math import fsum, log10
from statistics import median

ANALYSIS_BLOCKS = 10
MIN_ANALYSIS_SAMPLES = 30
STABLE_SPREAD_DB = 3.0
SEPARATION_DB = 4.0
QUIET_SPREAD_DB = 3.0
MIN_QUIET_BLOCKS = 3
MIN_QUIET_COVERAGE = 0.4


def _merge(spans):
    merged = []
    for begin, end in sorted(spans):
        if merged and merged[-1][1] == begin:
            merged[-1][1] = end
        else:
            merged.append([begin, end])
    return [tuple(span) for span in merged]


def _interval(signal, begin, end):
    rate = signal.rate
    lines = signal.line_numbers
    peak = signal.peak_index
    return {'from_s': begin / rate, 'to_s': end / rate,
            'from_line': lines[peak + begin], 'to_line': lines[peak + end - 1]}


def _intervals(signal, spans):
    return [_interval(signal, begin, end) for begin, end in _merge(spans)]


def _decide(means):
    """按十等分块均功率判定背景区间，返回 (decision, estimate, quiet, basis, summary)。"""
    centre = median(means)
    if centre <= 0:
        if max(means) > 0:
            return 'undecided', 0.0, [], {}, '尾段零功率块与有效块并存，无法界定背景，退回补测'
        return 'whole', 0.0, [], {}, '尾段功率为零，按零噪声处理'
    rel = [10 * log10(max(mean, centre * 1e-9) / centre) for mean in means]
    spread = max(rel) - min(rel)
    basis = {'spread_db': spread}
    if spread <= STABLE_SPREAD_DB:
        return ('whole', centre, [], basis,
                f'尾段 {ANALYSIS_BLOCKS} 等分块功率极差 {spread:.1f} dB '
                f'≤ {STABLE_SPREAD_DB:g} dB，按整段五段中位数估计')
    ordered = sorted(rel)
    gaps = [upper - lower for lower, upper in zip(ordered, ordered[1:])]
    widest = max(gaps)
    basis['separation_db'] = widest
    if widest < SEPARATION_DB:
        return ('whole', centre, [], basis,
                f'尾段块功率极差 {spread:.1f} dB，但相邻块最大间隔 {widest:.1f} dB '
                f'< {SEPARATION_DB:g} dB（衰减余量缓降，非启停起伏），按整段五段中位数估计')
    cut = ordered[gaps.index(widest)]
    quiet = [index for index in range(ANALYSIS_BLOCKS) if rel[index] <= cut]
    quiet_spread = max(rel[index] for index in quiet) - min(rel[index] for index in quiet)
    coverage = len(quiet) / ANALYSIS_BLOCKS
    basis.update(quiet_coverage=coverage, quiet_spread_db=quiet_spread)
    estimate = median(means[index] for index in quiet)
    if estimate <= 0:
        return 'undecided', 0.0, quiet, basis, '安静块功率为零，无法界定背景，退回补测'
    if len(quiet) < MIN_QUIET_BLOCKS or coverage < MIN_QUIET_COVERAGE:
        return ('undecided', estimate, quiet, basis,
                f'高低块分离 {widest:.1f} dB，但安静块仅 {len(quiet)}/{ANALYSIS_BLOCKS} 块'
                f'（覆盖 {coverage:.0%} < {MIN_QUIET_COVERAGE:.0%}），瞬时静音不能作为背景，退回补测')
    if quiet_spread > QUIET_SPREAD_DB:
        return ('undecided', estimate, quiet, basis,
                f'高低块分离 {widest:.1f} dB，但安静块自身极差 {quiet_spread:.1f} dB '
                f'> {QUIET_SPREAD_DB:g} dB，无法界定背景，退回补测')
    return ('segment', estimate, quiet, basis,
            f'高低块分离 {widest:.1f} dB ≥ {SEPARATION_DB:g} dB；安静块 {len(quiet)}/{ANALYSIS_BLOCKS} 块'
            f'（覆盖 {coverage:.0%}）自身极差 {quiet_spread:.1f} dB ≤ {QUIET_SPREAD_DB:g} dB，'
            '按平稳段中位数估计')


def assess_background(signal, power, tail_from, whole_noise):
    """界定测噪尾段的背景估计区间，给出说得清的依据。

    尾段十等分判定：平稳或仅缓变时按整段五段均方中位数；高低块可分离且安静块
    足够长、自身平稳时按平稳段中位数，高块区间标记为剔除起伏（仅标记，原始
    采样行与曲线点保留）；无法界定时 undecided，由调用方退回补测。整段估计值
    始终保留在 whole_tail_pa2 供对照。
    """
    rate = signal.rate
    length = len(power) - tail_from
    report = {'decision': 'whole', 'estimate_pa2': whole_noise, 'whole_tail_pa2': whole_noise,
                  'tail_from_s': tail_from / rate, 'tail_to_s': len(power) / rate,
                  'blocks_db': None, 'adopted': _intervals(signal, [(tail_from, len(power))]),
                  'excluded': [], 'basis': {'blocks': ANALYSIS_BLOCKS}, 'summary': ''}
    if length < MIN_ANALYSIS_SAMPLES:
        report['summary'] = f'尾段仅 {length} 点，不足 {MIN_ANALYSIS_SAMPLES} 点，按整段五段中位数估计'
        return report
    spans = [(tail_from + index * length // ANALYSIS_BLOCKS,
              tail_from + (index + 1) * length // ANALYSIS_BLOCKS) for index in range(ANALYSIS_BLOCKS)]
    means = [fsum(power[begin:end]) / (end - begin) for begin, end in spans]
    decision, estimate, quiet, basis, summary = _decide(means)
    report['basis'].update(basis)
    if decision == 'segment':
        loud = [index for index in range(ANALYSIS_BLOCKS) if index not in quiet]
        excluded = []
        for begin, end in _merge(spans[index] for index in loud):
            members = [means[index] for index, span in enumerate(spans) if begin <= span[0] and span[1] <= end]
            excess = 10 * log10(fsum(members) / len(members) / estimate)
            excluded.append({**_interval(signal, begin, end), 'excess_db': excess})
        report.update(estimate_pa2=estimate, excluded=excluded,
                      adopted=_intervals(signal, [spans[index] for index in quiet]))
        summary += f'，剔除 {len(excluded)} 段起伏，原始尾段记录保留'
    report['decision'] = decision
    report['summary'] = summary
    adopted = report['estimate_pa2']
    if adopted > 0:
        report['blocks_db'] = [10 * log10(max(mean, adopted * 1e-9) / adopted) for mean in means]
    return report
