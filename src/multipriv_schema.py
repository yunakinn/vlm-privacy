from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CHINESE_TO_CANONICAL = {
    "姓名": "Name",
    "证件号": "ID Number",
    "民族": "NATIONALITY",
    "性别": "Gender",
    "出生日期": "Birth",
    "住址": "Address",
    "银行卡号": "Bank Card Number",
    "航班时间": "Flight Time",
    "航班地点": "Flight Route",
    "班次": "Flight Number",
    "登机口": "Boarding Gate",
    "航站楼": "Terminal",
    "座位号": "Seat Number",
    "舱位": "Cabin Class",
    "车牌": "License Plate Number",
    "车辆类型": "Vehicle Type",
    "检查类型": "Examination Type",
    "检查内容": "Examination Findings",
    "诊断提示": "Diagnosis",
    "药物治疗": "Medication",
    "报告医生": "Reporting Physician",
    "财务状况": "Financial Status",
    "岗位工资": "Job Position Salary",
    "综合补贴": "Comprehensive Allowance",
    "年终奖": "Year-end Bonus",
    "同框关系": "Relationship in the Same Frame",
}


def detect_profile_language(profile: dict[str, Any]) -> str:
    if any(key in CHINESE_TO_CANONICAL for key in profile):
        return "zh"

    return "en"


def normalize_profile(
    profile: dict[str, Any],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}

    for key, value in profile.items():
        canonical_key = CHINESE_TO_CANONICAL.get(
            key,
            key,
        )
        normalized[canonical_key] = value

    normalized["_source_language"] = (
        detect_profile_language(profile)
    )

    return normalized


def load_person_profile(
    person_dir: str | Path,
) -> dict[str, Any]:
    directory = Path(person_dir)
    json_files = sorted(directory.glob("*.json"))

    if len(json_files) != 1:
        raise RuntimeError(
            f"Expected one JSON in {directory}, "
            f"found {len(json_files)}"
        )

    with json_files[0].open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if (
        not isinstance(data, list)
        or len(data) != 1
        or not isinstance(data[0], dict)
    ):
        raise ValueError(
            f"Unexpected profile format: {json_files[0]}"
        )

    profile = normalize_profile(data[0])
    profile["_json_file"] = json_files[0].name

    return profile
