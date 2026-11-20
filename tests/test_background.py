import pytest

from app.modules.m3_decay.integral import Curve, integrate, reverse_energy
from app.modules.m4_fit.service import fits
from app.modules.m5_noise.gate import noise_gate
from app.modules.m6_reconcile.comparison import compare
from app.reasons import reason


def ac_signal(signal, pattern, factor=0.996, quiet=0.01, loud=0.05, zero='0'):
    """850 点指数衰减 + 150 点尾段，尾段十块按 pattern 标记（. 安静 / # 空调开 / 0 零功率）。"""
    decay = [2 * factor ** index for index in range(850)]
    tail = []
    for mark in pattern:
        amplitude = {'.': quiet, '#': loud, zero: 0.0}[mark]
        tail += [amplitude] * 15
    return signal(tuple(decay + tail))


def gate_with(curve, noise):
    from math import log10
    energy = reverse_energy(curve.power, noise)
    initial = energy[0]
    db = [10 * log10(value / initial) if value > 0 else None for value in energy]
    floor = 10 * log10(noise / initial) if noise > 0 else None
    return noise_gate(Curve(curve.signal, curve.power, db, curve.tail_from, noise, floor, initial))


def test_sample_tails_use_whole_segment_with_legacy_estimate(report):
    backgrounds = {row['band_hz']: row['background'] for row in report['noise']['bands']}
    assert backgrounds and all(bg['decision'] == 'whole' for bg in backgrounds.values())
    assert all(bg['excluded'] == [] and bg['estimate_pa2'] == bg['whole_tail_pa2'] for bg in backgrounds.values())


def test_cycling_tail_uses_stable_segment_and_keeps_original_records(signal):
    curve = integrate(ac_signal(signal, '##...##...'))
    background = curve.background
    assert background['decision'] == 'segment'
    assert background['estimate_pa2'] == pytest.approx(1e-4, rel=1e-6)
    assert background['whole_tail_pa2'] > 10 * background['estimate_pa2']  # 整段会抬高 10 dB 以上
    assert [(span['from_s'], span['to_s']) for span in background['excluded']] == [(0.85, 0.88), (0.925, 0.955)]
    assert [(span['from_line'], span['to_line']) for span in background['excluded']] == [(852, 881), (927, 956)]
    assert all(span['excess_db'] == pytest.approx(13.98, abs=.01) for span in background['excluded'])
    assert [(span['from_s'], span['to_s']) for span in background['adopted']] == [(0.88, 0.925), (0.955, 1.0)]
    assert len(curve.power) == 1000 and len(curve.db) == 1000  # 剔除仅标记，原始尾段记录保留
    gate = noise_gate(curve)
    assert gate['decay_class'] == 'T30' and gate['usable_depth_db'] >= 35.0
    assert gate_with(curve, background['whole_tail_pa2'])['decay_class'] == 'T20'  # 整段估值只剩 T20


def test_momentary_silence_is_remeasure_not_background(signal):
    curve = integrate(ac_signal(signal, '#########.'))
    assert curve.background['decision'] == 'undecided'
    assert curve.error['code'] == 'NOISE_UNSTABLE'
    gate = noise_gate(curve)
    assert gate['decay_class'] == 'REMEASURE' and not gate['supports_t20'] and not gate['supports_t30']
    report, _ = fits({500: curve})
    trial = report['bands'][0]
    assert [trial[key]['error']['code'] for key in ('t20', 't30')] == ['NOISE_UNSTABLE'] * 2
    row = compare({'band_hz': 500, 't20': trial['t20'], 't30': trial['t30']})
    assert row['status'] == 'ISOLATED'
    assert any(item['code'] == 'NOISE_UNSTABLE' for item in row['findings'])


def test_brief_burst_is_excluded_without_losing_tail(signal):
    curve = integrate(ac_signal(signal, '.........#'))
    background = curve.background
    assert background['decision'] == 'segment' and len(background['excluded']) == 1
    assert (background['excluded'][0]['from_s'], background['excluded'][0]['to_s']) == (0.985, 1.0)
    assert background['estimate_pa2'] == pytest.approx(1e-4, rel=1e-6)


def test_late_single_step_uses_early_quiet_segment(signal):
    curve = integrate(ac_signal(signal, '.....#####'))
    background = curve.background
    assert background['decision'] == 'segment'
    assert [(span['from_s'], span['to_s']) for span in background['adopted']] == [(0.85, 0.925)]
    assert [(span['from_s'], span['to_s']) for span in background['excluded']] == [(0.925, 1.0)]


def test_zero_power_blocks_force_remeasure(signal):
    curve = integrate(ac_signal(signal, '...0...0..'))
    assert curve.background['decision'] == 'undecided'
    assert curve.error['code'] == 'NOISE_UNSTABLE'


def test_gradual_drift_tail_stays_whole(signal):
    values = tuple([2 * 0.996 ** index for index in range(850)] +
                   [0.01 * 0.97 ** index for index in range(150)])
    curve = integrate(signal(values))
    background = curve.background
    assert background['decision'] == 'whole' and background['basis']['spread_db'] > 3.0
    assert background['basis']['separation_db'] < 4.0 and background['excluded'] == []


def test_same_background_across_decay_noise_fit_and_archive(report):
    decay_bg = {row['band_hz']: row['background'] for row in report['decay']['bands']}
    noise_bg = {row['band_hz']: row['background'] for row in report['noise']['bands']}
    assert decay_bg == noise_bg  # 曲线与噪声判定同源
    estimates = {row['band_hz']: row['noise_power_pa2'] for row in report['noise']['bands']}
    assert estimates == {row['band_hz']: row['noise_power_pa2'] for row in report['decay']['bands']}


def test_archive_rerun_keeps_single_entry_with_same_background(client, source):
    preview = client.get(f'/api/reconcile/{source}/preview').json()
    run = client.post('/api/runs', json={'source_id': source}).json()
    again = client.post('/api/runs', json={'source_id': source}).json()
    assert again['reused'] and again['id'] == run['id']
    assert client.get('/api/runs').json()['runs'] == [run['manifest']]
    assert run['result']['noise'] == preview['noise'] and run['result']['decay'] == preview['decay']
    retry = client.post(f"/api/runs/{run['id']}/rerun").json()
    assert retry['reused'] and retry['result']['noise'] == preview['noise']
    assert client.get('/api/runs').json()['runs'] == [run['manifest']]


def test_remeasure_reason_is_registered():
    assert reason('NOISE_UNSTABLE')['message'] == '尾段背景起伏无法界定，退回补测'
