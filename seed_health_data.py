# -*- coding: utf-8 -*-
"""Seed activity data for local UI/API testing."""

import argparse
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(r"C:\fall-detection\health_log.db")

SCENARIOS = {
    "normal_7days": {
        "description": "정상 7일 데이터",
        "days": 7,
        "points_per_day": 480,
        "posture_dist": {"standing": 0.35, "sitting": 0.30, "lying": 0.20, "unknown": 0.15},
        "abnormal_rate": 0.05,
    },
    "activity_drop": {
        "description": "정상 7일 + 오늘 활동성 저하",
        "days": 8,
        "points_per_day": 480,
        "posture_dist": {"standing": 0.35, "sitting": 0.30, "lying": 0.20, "unknown": 0.15},
        "today_override": {"standing": 0.10, "sitting": 0.20, "lying": 0.55, "unknown": 0.15},
        "abnormal_rate": 0.05,
    },
    "fall_today": {
        "description": "정상 7일 + 오늘 낙상 1건",
        "days": 8,
        "points_per_day": 480,
        "posture_dist": {"standing": 0.35, "sitting": 0.30, "lying": 0.20, "unknown": 0.15},
        "today_fall": 1,
        "abnormal_rate": 0.05,
    },
    "camera_issue": {
        "description": "카메라 인식 실패가 많은 시나리오",
        "days": 3,
        "points_per_day": 480,
        "posture_dist": {"standing": 0.10, "sitting": 0.10, "lying": 0.10, "unknown": 0.70},
        "abnormal_rate": 0.05,
        "no_landmark_rate": 0.65,
    },
    "posture_unstable": {
        "description": "자세 불안정 증가",
        "days": 7,
        "points_per_day": 480,
        "posture_dist": {"standing": 0.30, "sitting": 0.40, "lying": 0.20, "unknown": 0.10},
        "abnormal_rate": 0.30,
    },
    "baseline_14days": {
        "description": "개인 기준선용 안정 14일 데이터",
        "days": 14,
        "points_per_day": 480,
        "posture_dist": {"standing": 0.30, "sitting": 0.35, "lying": 0.25, "unknown": 0.10},
        "abnormal_rate": 0.05,
        "slot_aware": True,
    },
    "fall_pattern": {
        "description": "낙상 전후 패턴 분석용 데이터",
        "days": 10,
        "points_per_day": 480,
        "posture_dist": {"standing": 0.30, "sitting": 0.35, "lying": 0.25, "unknown": 0.10},
        "today_fall": 1,
        "abnormal_rate": 0.05,
        "decline_last_n_days": 3,
        "decline_severity": 0.4,
    },
    "night_activity": {
        "description": "밤 시간 활동 증가 시나리오",
        "days": 7,
        "points_per_day": 480,
        "posture_dist": {"standing": 0.30, "sitting": 0.35, "lying": 0.25, "unknown": 0.10},
        "abnormal_rate": 0.05,
        "night_active": True,
    },
}


def _weighted_choice(dist):
    value = random.random()
    cumulative = 0
    for key, weight in dist.items():
        cumulative += weight
        if value < cumulative:
            return key
    return list(dist.keys())[-1]


def _slot_distribution(hour, base_dist, night_active=False):
    if 22 <= hour or hour < 6:
        if night_active:
            return {"standing": 0.20, "sitting": 0.30, "lying": 0.40, "unknown": 0.10}
        return {"standing": 0.02, "sitting": 0.03, "lying": 0.90, "unknown": 0.05}
    if 6 <= hour < 12:
        return {"standing": 0.45, "sitting": 0.35, "lying": 0.10, "unknown": 0.10}
    if 12 <= hour < 18:
        return {"standing": 0.30, "sitting": 0.45, "lying": 0.15, "unknown": 0.10}
    if 18 <= hour < 22:
        return {"standing": 0.15, "sitting": 0.45, "lying": 0.30, "unknown": 0.10}
    return base_dist


def _apply_decline(dist, severity):
    standing = dist.get("standing", 0.3) * (1 - severity)
    sitting = dist.get("sitting", 0.35) * (1 - severity)
    extra_lying = (dist.get("standing", 0.3) - standing) + (dist.get("sitting", 0.35) - sitting)
    return {
        "standing": standing,
        "sitting": sitting,
        "lying": dist.get("lying", 0.25) + extra_lying,
        "unknown": dist.get("unknown", 0.10),
    }


def seed_scenario(scenario_name):
    cfg = SCENARIOS[scenario_name]
    print(f"\n[seed] {scenario_name}: {cfg['description']}")
    print(f"  days={cfg['days']}, points_per_day={cfg['points_per_day']}")

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    today = datetime.now().date()
    inserted = 0
    fall_events_added = 0

    for offset in range(cfg["days"]):
        day = today - timedelta(days=cfg["days"] - 1 - offset)
        is_today = day == today
        days_to_today = (today - day).days
        dist = cfg.get("today_override") if (is_today and "today_override" in cfg) else cfg["posture_dist"]

        if cfg.get("decline_last_n_days") and days_to_today <= cfg["decline_last_n_days"]:
            dist = _apply_decline(dist, cfg.get("decline_severity", 0.3))

        slot_aware = cfg.get("slot_aware", False) or cfg.get("night_active", False)
        minutes_in_period = 24 * 60 if slot_aware else 8 * 60
        start_hour = 0 if slot_aware else 9

        for i in range(cfg["points_per_day"]):
            minute_offset = int(i * (minutes_in_period / cfg["points_per_day"]))
            ts = datetime.combine(day, datetime.min.time()) + timedelta(hours=start_hour, minutes=minute_offset)
            slot_dist = _slot_distribution(ts.hour, dist, cfg.get("night_active", False)) if slot_aware else dist
            posture = _weighted_choice(slot_dist)
            abnormal = 1 if random.random() < cfg.get("abnormal_rate", 0.05) else 0
            stillness = random.uniform(0, 15) if posture == "lying" else random.uniform(0, 3)
            no_landmark_rate = cfg.get("no_landmark_rate", 0.05)
            landmark_detected = 0 if random.random() < no_landmark_rate else 1

            c.execute("""
                INSERT INTO activity_log
                (timestamp, date, hour, posture, fall_detected, score,
                 xgb_proba, abnormal_posture, stillness_sec, rule_pos, xgb_pos,
                 source, landmark_detected)
                VALUES (?, ?, ?, ?, 0, 0, ?, ?, ?, 0, 0, 'seed', ?)
            """, (
                ts.isoformat(),
                day.strftime("%Y-%m-%d"),
                ts.hour,
                posture,
                random.uniform(0, 0.3),
                abnormal,
                stillness,
                landmark_detected,
            ))
            inserted += 1

        if is_today and cfg.get("today_fall", 0) > 0:
            for _ in range(cfg["today_fall"]):
                fall_ts = datetime.combine(day, datetime.min.time()) + timedelta(hours=14, minutes=23)
                c.execute("""
                    INSERT INTO fall_event_log
                    (timestamp, date, posture, score, xgb_proba, ensemble_mode, capture_filename, source)
                    VALUES (?, ?, 'lying', 5, 0.92, 'and', 'seed_fall.jpg', 'seed')
                """, (fall_ts.isoformat(), day.strftime("%Y-%m-%d")))
                fall_events_added += 1

    conn.commit()
    conn.close()
    print(f"  inserted activity_log={inserted}, fall_event_log={fall_events_added}")


def clear_all(only_seed=True):
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    if only_seed:
        c.execute("DELETE FROM activity_log WHERE source='seed'")
        activity_deleted = c.rowcount
        c.execute("DELETE FROM fall_event_log WHERE source='seed'")
        fall_deleted = c.rowcount
        print(f"[clear] seed only: activity_log={activity_deleted}, fall_event_log={fall_deleted}")
    else:
        c.execute("DELETE FROM activity_log")
        activity_deleted = c.rowcount
        c.execute("DELETE FROM fall_event_log")
        fall_deleted = c.rowcount
        print(f"[clear] all: activity_log={activity_deleted}, fall_event_log={fall_deleted}")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed health activity data for testing")
    parser.add_argument("--scenario", default="normal_7days", choices=list(SCENARIOS.keys()))
    parser.add_argument("--clear", action="store_true", help="Delete only seed data")
    parser.add_argument("--clear-all", action="store_true", help="Delete all data including live data")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"[error] DB not found: {DB_PATH}. Run the FastAPI server or health_logger.init_db() first.")
        raise SystemExit(1)

    if args.clear_all:
        clear_all(only_seed=False)
    elif args.clear:
        clear_all(only_seed=True)
    else:
        seed_scenario(args.scenario)
        print("Next: call /health/activity/today, /trend, /slots, /baseline, or /fall-pattern")
