# -*- coding: utf-8 -*-
"""Activity logging and summary APIs for the fall-detection service.

This module does not change the fall-detection model. It only stores model
outputs and summarizes them as non-medical activity condition indicators.
"""

import sqlite3
import statistics
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(r"C:\fall-detection\health_log.db")
LOG_INTERVAL = 60
MIN_POINTS_FOR_SCORE = 30
MIN_POINTS_RELIABLE = 120
DAILY_VALID_POINTS = 120
BASELINE_DAYS = 14
BASELINE_MIN_VALID_DAYS = 10
FALL_DEDUP_SECONDS = 10

SCORE_LABELS = {
    "activity": "활동성",
    "stability": "안정성",
    "rest_balance": "휴식 균형",
    "posture_quality": "자세 상태",
    "safety": "안전도",
}

TIME_SLOTS = {
    "morning": (6, 12),
    "afternoon": (12, 18),
    "evening": (18, 22),
    "night": (22, 6),
}

SLOT_LABELS = {
    "morning": "오전 (06-12시)",
    "afternoon": "오후 (12-18시)",
    "evening": "저녁 (18-22시)",
    "night": "밤 (22-06시)",
}

_lock = threading.Lock()


def _connect():
    return sqlite3.connect(str(DB_PATH))


def _ensure_column(cursor, table, column, definition):
    cursor.execute(f"PRAGMA table_info({table})")
    columns = {row[1] for row in cursor.fetchall()}
    if column not in columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db():
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS activity_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT NOT NULL,
            date             TEXT NOT NULL,
            hour             INTEGER NOT NULL,
            posture          TEXT,
            fall_detected    INTEGER,
            score            INTEGER,
            xgb_proba        REAL,
            abnormal_posture INTEGER,
            stillness_sec    REAL,
            rule_pos         INTEGER,
            xgb_pos          INTEGER,
            source           TEXT DEFAULT 'live',
            landmark_detected INTEGER DEFAULT 1
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS fall_event_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT NOT NULL,
            date             TEXT NOT NULL,
            posture          TEXT,
            score            INTEGER,
            xgb_proba        REAL,
            ensemble_mode    TEXT,
            capture_filename TEXT,
            confirmed        INTEGER DEFAULT 0,
            source           TEXT DEFAULT 'live'
        )
    """)
    _ensure_column(c, "activity_log", "source", "TEXT DEFAULT 'live'")
    _ensure_column(c, "activity_log", "landmark_detected", "INTEGER DEFAULT 1")
    _ensure_column(c, "fall_event_log", "source", "TEXT DEFAULT 'live'")
    _ensure_column(c, "fall_event_log", "confirmed", "INTEGER DEFAULT 0")
    c.execute("CREATE INDEX IF NOT EXISTS idx_activity_date ON activity_log(date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_activity_timestamp ON activity_log(timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_activity_date_hour ON activity_log(date, hour)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_fall_date ON fall_event_log(date)")
    conn.commit()
    conn.close()
    print(f"[health_logger] DB 초기화 완료: {DB_PATH}")


def log_snapshot(status_dict):
    try:
        now = datetime.now()
        with _lock:
            conn = _connect()
            c = conn.cursor()
            c.execute("""
                INSERT INTO activity_log
                (timestamp, date, hour, posture, fall_detected, score,
                 xgb_proba, abnormal_posture, stillness_sec, rule_pos, xgb_pos,
                 source, landmark_detected)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'live', ?)
            """, (
                now.isoformat(),
                now.strftime("%Y-%m-%d"),
                now.hour,
                status_dict.get("posture", "unknown"),
                1 if status_dict.get("fall_detected") else 0,
                status_dict.get("score", 0),
                status_dict.get("xgb_proba", 0.0),
                1 if status_dict.get("abnormal_posture") else 0,
                status_dict.get("stillness_sec", 0.0),
                1 if status_dict.get("rule_pos") else 0,
                1 if status_dict.get("xgb_pos") else 0,
                1 if status_dict.get("landmark_detected", True) else 0,
            ))
            conn.commit()
            conn.close()
    except Exception as exc:
        print(f"[health_logger] 저장 실패: {exc}")


def start_logger_thread(status_ref):
    def loop():
        while True:
            log_snapshot(status_ref)
            time.sleep(LOG_INTERVAL)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    print(f"[health_logger] 백그라운드 로깅 시작 ({LOG_INTERVAL}초 간격)")


def log_fall_event(status_dict, capture_filename=None):
    try:
        now = datetime.now()
        cutoff = (now - timedelta(seconds=FALL_DEDUP_SECONDS)).isoformat()
        with _lock:
            conn = _connect()
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM fall_event_log WHERE timestamp >= ?", (cutoff,))
            if c.fetchone()[0]:
                conn.close()
                return False
            c.execute("""
                INSERT INTO fall_event_log
                (timestamp, date, posture, score, xgb_proba, ensemble_mode, capture_filename, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'live')
            """, (
                now.isoformat(),
                now.strftime("%Y-%m-%d"),
                status_dict.get("posture", "unknown"),
                status_dict.get("score", 0),
                status_dict.get("xgb_proba", 0.0),
                status_dict.get("ensemble_mode", ""),
                capture_filename,
            ))
            conn.commit()
            conn.close()
            return True
    except Exception as exc:
        print(f"[health_logger] 낙상 이벤트 저장 실패: {exc}")
        return False


def get_fall_events(days=7):
    cutoff = (datetime.now().date() - timedelta(days=max(0, days - 1))).strftime("%Y-%m-%d")
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        SELECT timestamp, posture, score, xgb_proba, ensemble_mode, capture_filename, confirmed, source
        FROM fall_event_log
        WHERE date >= ?
        ORDER BY timestamp DESC
    """, (cutoff,))
    cols = ["timestamp", "posture", "score", "xgb_proba", "ensemble_mode", "capture_filename", "confirmed", "source"]
    events = [dict(zip(cols, row)) for row in c.fetchall()]
    conn.close()
    return {"days": days, "count": len(events), "events": events}


def _build_where(date_str, hour_from=None, hour_to=None):
    where = "date=?"
    params = [date_str]
    if hour_from is not None and hour_to is not None:
        if hour_from < hour_to:
            where += " AND hour >= ? AND hour < ?"
            params += [hour_from, hour_to]
        else:
            where += " AND (hour >= ? OR hour < ?)"
            params += [hour_from, hour_to]
    return where, params


def _summary_for_date(date_str, hour_from=None, hour_to=None):
    conn = _connect()
    c = conn.cursor()
    where, params = _build_where(date_str, hour_from, hour_to)
    c.execute(f"SELECT posture, COUNT(*) FROM activity_log WHERE {where} GROUP BY posture", params)
    posture_dist = {row[0]: row[1] for row in c.fetchall()}
    c.execute(f"SELECT COUNT(*) FROM activity_log WHERE {where} AND fall_detected=1", params)
    fall_count = c.fetchone()[0]
    c.execute(f"SELECT COUNT(*) FROM activity_log WHERE {where} AND abnormal_posture=1", params)
    abnormal_count = c.fetchone()[0]
    c.execute(f"SELECT COUNT(*) FROM activity_log WHERE {where} AND stillness_sec >= 10", params)
    long_stillness = c.fetchone()[0]
    c.execute(f"SELECT COUNT(*) FROM activity_log WHERE {where}", params)
    total = c.fetchone()[0]
    c.execute(f"SELECT COUNT(*) FROM activity_log WHERE {where} AND landmark_detected=0", params)
    no_landmark_count = c.fetchone()[0]
    c.execute(f"SELECT posture FROM activity_log WHERE {where} ORDER BY timestamp", params)
    postures = [row[0] for row in c.fetchall()]
    conn.close()
    transitions = sum(1 for i in range(1, len(postures)) if postures[i] != postures[i - 1])
    return {
        "date": date_str,
        "total_minutes": total,
        "posture_distribution": posture_dist,
        "fall_count": fall_count,
        "abnormal_count": abnormal_count,
        "long_stillness_count": long_stillness,
        "posture_transitions": transitions,
        "no_landmark_count": no_landmark_count,
    }


def get_today_summary():
    return _summary_for_date(datetime.now().strftime("%Y-%m-%d"))


def _data_quality(total_points, unknown_points, no_landmark_points=0):
    valid_ratio = (total_points - unknown_points) / total_points if total_points else 0
    no_landmark_ratio = no_landmark_points / total_points if total_points else 0
    if total_points < MIN_POINTS_FOR_SCORE:
        level = "insufficient"
        message = "활동 데이터를 수집하는 중입니다. 충분한 기록이 쌓이면 지표가 표시됩니다."
    elif total_points < MIN_POINTS_RELIABLE:
        level = "low" if valid_ratio < 0.5 else "medium"
        message = "오늘 수집된 데이터가 적어 참고용으로만 활용하세요."
    elif no_landmark_ratio > 0.4:
        level = "camera_issue"
        message = "카메라가 사람을 잘 인식하지 못했어요. 카메라 위치, 조명, 가림을 확인해 주세요."
    elif valid_ratio < 0.5:
        level = "low"
        message = "오늘 자세 인식이 모호한 시간이 많았어요."
    elif valid_ratio < 0.75:
        level = "medium"
        message = "일부 구간에서 인식이 불안정했지만 참고 지표로 활용할 수 있습니다."
    else:
        level = "good"
        message = "오늘 데이터는 안정적으로 수집되었습니다."
    if no_landmark_ratio > 0.4:
        issue_type = "camera"
    elif valid_ratio < 0.5:
        issue_type = "posture"
    else:
        issue_type = "none"
    return {
        "valid_ratio": round(valid_ratio, 2),
        "data_points": total_points,
        "no_landmark_ratio": round(no_landmark_ratio, 2),
        "no_landmark_points": no_landmark_points,
        "issue_type": issue_type,
        "level": level,
        "message": message,
    }


def _score_from_summary(summary):
    total = max(1, summary["total_minutes"])
    dist = summary["posture_distribution"]
    standing = dist.get("standing", 0)
    sitting = dist.get("sitting", 0)
    lying = dist.get("lying", 0)
    unknown = dist.get("unknown", 0)
    active_ratio = (standing + sitting) / total
    transition_score = min(100, summary["posture_transitions"] * 2)
    activity = round((active_ratio * 100 * 0.6) + (transition_score * 0.4), 1)
    stability = max(0, round(100 - min(100, summary["fall_count"] * 20), 1))
    lying_ratio = lying / total
    if 0.10 <= lying_ratio <= 0.30:
        rest_balance = 100.0
    elif lying_ratio < 0.10:
        rest_balance = round(80 - (0.10 - lying_ratio) * 200, 1)
    else:
        rest_balance = round(100 - (lying_ratio - 0.30) * 150, 1)
    rest_balance = max(0, min(100, rest_balance))
    unknown_ratio = unknown / total
    abnormal_ratio = summary["abnormal_count"] / total
    posture_quality = max(0, round(100 - unknown_ratio * 60 - abnormal_ratio * 80, 1))
    safety_penalty = summary["fall_count"] * 25 + summary["long_stillness_count"] * 0.5
    safety = max(0, round(100 - safety_penalty, 1))
    return {
        "activity": activity,
        "stability": stability,
        "rest_balance": rest_balance,
        "posture_quality": posture_quality,
        "safety": safety,
    }


def calc_today_scores():
    summary = get_today_summary()
    total = summary["total_minutes"]
    unknown = summary["posture_distribution"].get("unknown", 0)
    no_landmark = summary.get("no_landmark_count", 0)
    quality = _data_quality(total, unknown, no_landmark)
    if total < MIN_POINTS_FOR_SCORE:
        return {
            "status": "collecting",
            "date": summary["date"],
            "data_quality": quality,
            "raw": summary,
            "message": quality["message"],
        }
    scores = _score_from_summary(summary)
    overall = round(sum(scores.values()) / len(scores), 1)
    return {
        "status": "ok",
        "date": summary["date"],
        "data_points": total,
        "scores": scores,
        "labels": SCORE_LABELS,
        "overall": overall,
        "overall_note": "참고용 활동 컨디션 요약 (건강 점수 아님)",
        "data_quality": quality,
        "raw": summary,
    }


def _valid_past_days(days, min_points=DAILY_VALID_POINTS):
    today = datetime.now().strftime("%Y-%m-%d")
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        SELECT date, COUNT(*) AS cnt
        FROM activity_log
        WHERE date != ?
        GROUP BY date
        HAVING cnt >= ?
        ORDER BY date DESC
    """, (today, min_points))
    result = [row[0] for row in c.fetchall()][:days]
    conn.close()
    return result


def _calc_past_average(dates):
    if not dates:
        return {key: 0 for key in SCORE_LABELS}
    accum = {key: 0 for key in SCORE_LABELS}
    used = 0
    for date_str in dates:
        summary = _summary_for_date(date_str)
        if summary["total_minutes"] < MIN_POINTS_FOR_SCORE:
            continue
        scores = _score_from_summary(summary)
        for key, value in scores.items():
            accum[key] += value
        used += 1
    if used == 0:
        return {key: 0 for key in SCORE_LABELS}
    return {key: round(value / used, 1) for key, value in accum.items()}


def get_trend(days=7):
    past_days = _valid_past_days(days)
    if len(past_days) < days:
        return {
            "status": "building_baseline",
            "past_valid_days_collected": len(past_days),
            "past_valid_days_needed": days,
            "min_points_per_day": DAILY_VALID_POINTS,
            "message": f"활동 기준선을 만드는 중입니다. 현재 과거 {len(past_days)}일치 유효 데이터가 쌓였어요.",
        }
    today_scores = calc_today_scores()
    if today_scores.get("status") != "ok":
        return {
            "status": "today_insufficient",
            "past_valid_days_collected": len(past_days),
            "today": today_scores,
            "message": "과거 데이터는 충분하지만 오늘 데이터가 부족해서 비교할 수 없습니다.",
        }
    past_avg = _calc_past_average(past_days)
    changes = {}
    for key, today_value in today_scores["scores"].items():
        past_value = past_avg.get(key, today_value)
        diff = today_value - past_value
        pct = (diff / past_value * 100) if past_value > 0 else 0
        changes[key] = {
            "today": today_value,
            "past_7day_avg": round(past_value, 1),
            "diff": round(diff, 1),
            "pct_change": round(pct, 1),
        }
    alerts = []
    for key, change in changes.items():
        label = SCORE_LABELS.get(key, key)
        if change["pct_change"] <= -20:
            alerts.append(f"{label} 점수가 평소보다 {abs(change['pct_change']):.0f}% 낮아졌어요.")
        elif change["pct_change"] >= 20:
            alerts.append(f"{label} 점수가 평소보다 {change['pct_change']:.0f}% 높아졌어요.")
    return {
        "status": "ok",
        "past_valid_days_collected": len(past_days),
        "today_scores": today_scores,
        "past_7day_avg_scores": past_avg,
        "labels": SCORE_LABELS,
        "changes": changes,
        "alerts": alerts if alerts else ["평소와 비슷한 패턴입니다."],
    }


def get_today_slots():
    today = datetime.now().strftime("%Y-%m-%d")
    slots = {}
    for name, (hour_from, hour_to) in TIME_SLOTS.items():
        summary = _summary_for_date(today, hour_from, hour_to)
        if summary["total_minutes"] < 10:
            slots[name] = {
                "status": "no_data",
                "label": SLOT_LABELS[name],
                "hours": f"{hour_from:02d}-{hour_to:02d}",
                "data_points": summary["total_minutes"],
            }
            continue
        scores = _score_from_summary(summary)
        total = max(1, summary["total_minutes"])
        dist_pct = {key: round(value / total * 100, 1) for key, value in summary["posture_distribution"].items()}
        slots[name] = {
            "status": "ok",
            "label": SLOT_LABELS[name],
            "hours": f"{hour_from:02d}-{hour_to:02d}",
            "data_points": summary["total_minutes"],
            "scores": scores,
            "posture_distribution_pct": dist_pct,
            "fall_count": summary["fall_count"],
        }
    return {
        "date": today,
        "labels": SLOT_LABELS,
        "score_labels": SCORE_LABELS,
        "slots": slots,
    }


def calc_personal_baseline(days=BASELINE_DAYS):
    today = datetime.now().strftime("%Y-%m-%d")
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        SELECT date, COUNT(*)
        FROM activity_log
        WHERE date > ? AND date != ?
        GROUP BY date
        HAVING COUNT(*) >= ?
        ORDER BY date DESC
    """, (cutoff, today, DAILY_VALID_POINTS))
    valid_days = [row[0] for row in c.fetchall()]
    conn.close()
    if len(valid_days) < BASELINE_MIN_VALID_DAYS:
        return {
            "status": "building_baseline",
            "valid_days_collected": len(valid_days),
            "valid_days_needed": BASELINE_MIN_VALID_DAYS,
            "lookback_days": days,
            "message": f"개인 기준선을 만드는 중입니다. 최근 {days}일 중 {len(valid_days)}일이 유효합니다.",
        }
    daily_scores = {key: [] for key in SCORE_LABELS}
    for date_str in valid_days:
        scores = _score_from_summary(_summary_for_date(date_str))
        for key in daily_scores:
            daily_scores[key].append(scores[key])
    baseline = {}
    for key, values in daily_scores.items():
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) >= 2 else 0
        baseline[key] = {
            "mean": round(mean, 1),
            "stdev": round(stdev, 1),
            "min": round(min(values), 1),
            "max": round(max(values), 1),
            "n": len(values),
        }
    today_scores = calc_today_scores()
    today_comparison = None
    if today_scores.get("status") == "ok":
        today_comparison = {}
        for key, value in today_scores["scores"].items():
            mean = baseline[key]["mean"]
            stdev = baseline[key]["stdev"]
            z_score = (value - mean) / stdev if stdev > 0 else 0
            level = "anomaly" if abs(z_score) >= 2 else "deviation" if abs(z_score) >= 1 else "normal"
            today_comparison[key] = {
                "today": value,
                "personal_mean": mean,
                "z_score": round(z_score, 2),
                "level": level,
            }
    return {
        "status": "ok",
        "valid_days_collected": len(valid_days),
        "lookback_days": days,
        "baseline": baseline,
        "labels": SCORE_LABELS,
        "today_comparison": today_comparison,
        "note": "개인 평소 패턴 기준 비교입니다. 진단이 아닌 참고 지표입니다.",
    }


def _avg_scores_in_range(start_date, end_date):
    conn = _connect()
    c = conn.cursor()
    c.execute("""
        SELECT date, COUNT(*)
        FROM activity_log
        WHERE date >= ? AND date <= ?
        GROUP BY date
        HAVING COUNT(*) >= ?
    """, (start_date.isoformat(), end_date.isoformat(), MIN_POINTS_FOR_SCORE))
    valid_days = [row[0] for row in c.fetchall()]
    conn.close()
    if not valid_days:
        return None
    accum = {key: 0 for key in SCORE_LABELS}
    for date_str in valid_days:
        scores = _score_from_summary(_summary_for_date(date_str))
        for key in accum:
            accum[key] += scores[key]
    return {key: round(value / len(valid_days), 1) for key, value in accum.items()}


def analyze_fall_pattern():
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT timestamp, date, capture_filename FROM fall_event_log ORDER BY timestamp DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    if not row:
        return {
            "status": "no_fall_event",
            "message": "최근 낙상 이벤트가 없습니다. 낙상 발생 후 전후 패턴을 분석할 수 있습니다.",
        }
    fall_ts, fall_date, capture_filename = row
    fall_day = datetime.fromisoformat(fall_ts).date()
    baseline_start = fall_day - timedelta(days=7)
    baseline_end = fall_day - timedelta(days=4)
    recent_start = fall_day - timedelta(days=3)
    recent_end = fall_day - timedelta(days=1)
    baseline_scores = _avg_scores_in_range(baseline_start, baseline_end)
    recent_scores = _avg_scores_in_range(recent_start, recent_end)
    if baseline_scores is None or recent_scores is None:
        return {
            "status": "insufficient_history",
            "fall_event": {"timestamp": fall_ts, "date": fall_date, "capture": capture_filename},
            "message": "낙상 전 기간 데이터가 부족해서 전후 패턴을 분석할 수 없습니다.",
        }
    changes = {}
    warning_signs = []
    for key in SCORE_LABELS:
        before = baseline_scores.get(key, 0)
        recent = recent_scores.get(key, 0)
        diff = recent - before
        pct = (diff / before * 100) if before > 0 else 0
        changes[key] = {
            "baseline_7d": round(before, 1),
            "recent_3d": round(recent, 1),
            "diff": round(diff, 1),
            "pct_change": round(pct, 1),
        }
        if pct <= -15:
            warning_signs.append(f"낙상 전 {SCORE_LABELS[key]} 점수가 평소보다 {abs(pct):.0f}% 감소했어요.")
    return {
        "status": "ok",
        "fall_event": {"timestamp": fall_ts, "date": fall_date, "capture": capture_filename},
        "baseline_period": {"start": baseline_start.isoformat(), "end": baseline_end.isoformat()},
        "recent_period": {"start": recent_start.isoformat(), "end": recent_end.isoformat()},
        "changes": changes,
        "labels": SCORE_LABELS,
        "warning_signs": warning_signs if warning_signs else ["낙상 직전 활동 패턴에서 큰 변화는 관찰되지 않았습니다."],
    }
