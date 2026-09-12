from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Any, List, Tuple

class FinancialForecaster:
    def __init__(self, horizon_days: int = 90):
        self.horizon_days = horizon_days

    def _parse_date(self, d_str: str) -> datetime:
        return datetime.strptime(str(d_str).strip(), "%Y-%m-%d")

    def simulate_balance(
        self,
        current_balance: float,
        min_balance: float,
        start_date: datetime,
        events: List[Dict[str, Any]],
        extra_deductions: Dict[str, float] = None
    ) -> Tuple[bool, float, List[Dict[str, Any]]]:
        """
        Simulates the daily balance over 90 days.
        Returns:
            - is_valid (bool): True if balance stays >= min_balance every single day.
            - min_projected_balance (float): Lowest balance recorded during the 90 days.
            - timeline (list): Daily projected balance log.
        """
        extra_deductions = extra_deductions or {}
        running_balance = current_balance
        min_seen = running_balance
        timeline = []

        # Map active resolved events to dates
        daily_deltas: Dict[str, float] = {}
        for ev in events:
            date_str = str(ev.get("date"))
            amt = float(ev.get("amount", 0.0))
            ev_type = ev.get("type", "").lower()

            if ev_type in ["income", "credit", "refund"]:
                daily_deltas[date_str] = daily_deltas.get(date_str, 0.0) + amt
            elif ev_type in ["expense", "debit"]:
                daily_deltas[date_str] = daily_deltas.get(date_str, 0.0) - amt

        is_valid = True
        for day_offset in range(self.horizon_days + 1):
            cur_date = start_date + timedelta(days=day_offset)
            date_key = cur_date.strftime("%Y-%m-%d")

            # Apply events scheduled for this day
            running_balance += daily_deltas.get(date_key, 0.0)

            # Apply planned payment deductions
            if date_key in extra_deductions:
                running_balance -= extra_deductions[date_key]

            if running_balance < min_balance:
                is_valid = False

            if running_balance < min_seen:
                min_seen = running_balance

            timeline.append({"date": date_key, "balance": running_balance})

        return is_valid, min_seen, timeline

    def get_amount_safe_to_pay(
        self,
        current_balance: float,
        min_balance: float,
        request_date: str,
        events: List[Dict[str, Any]]
    ) -> float:
        """
        Calculates the maximum amount payable on request_date such that 
        the balance never dips below min_balance over the 90-day window.
        """
        start_dt = self._parse_date(request_date)
        _, min_projected, _ = self.simulate_balance(
            current_balance=current_balance,
            min_balance=min_balance,
            start_date=start_dt,
            events=events
        )
        safe_amount = max(0.0, min_projected - min_balance)
        return round(safe_amount, 2)

    def get_earliest_date_for_full_payment(
        self,
        current_balance: float,
        min_balance: float,
        request_date: str,
        full_amount: float,
        events: List[Dict[str, Any]]
    ) -> str:
        """
        Finds the earliest date within the 90-day horizon where paying 
        the full amount preserves the minimum balance requirement throughout.
        """
        start_dt = self._parse_date(request_date)

        for day_offset in range(self.horizon_days + 1):
            candidate_dt = start_dt + timedelta(days=day_offset)
            candidate_str = candidate_dt.strftime("%Y-%m-%d")

            is_safe, _, _ = self.simulate_balance(
                current_balance=current_balance,
                min_balance=min_balance,
                start_date=start_dt,
                events=events,
                extra_deductions={candidate_str: full_amount}
            )

            if is_safe:
                return candidate_str

        return "none"
