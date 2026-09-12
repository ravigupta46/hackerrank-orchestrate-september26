import os
import pandas as pd
from typing import Dict, Any, Optional

DATASET_DIR = "dataset"
MEDIA_DIR = os.path.join(DATASET_DIR, "media", "images")

class DataLoader:
    def __init__(self, data_dir: str = DATASET_DIR):
        self.data_dir = data_dir
        self.profiles_df: Optional[pd.DataFrame] = None
        self.events_df: Optional[pd.DataFrame] = None
        self.requests_df: Optional[pd.DataFrame] = None
        self.rates_df: Optional[pd.DataFrame] = None
        self.options_df: Optional[pd.DataFrame] = None
        self.messages_df: Optional[pd.DataFrame] = None
        self.images_df: Optional[pd.DataFrame] = None

    def load_all(self):
        """Loads all base CSV files into memory."""
        self.profiles_df = pd.read_csv(os.path.join(self.data_dir, "financial_profiles.csv"))
        self.events_df = pd.read_csv(os.path.join(self.data_dir, "financial_events.csv"))
        self.requests_df = pd.read_csv(os.path.join(self.data_dir, "requests.csv"))
        self.rates_df = pd.read_csv(os.path.join(self.data_dir, "exchange_rates.csv"))
        self.options_df = pd.read_csv(os.path.join(self.data_dir, "request_payment_options.csv"))
        self.messages_df = pd.read_csv(os.path.join(self.data_dir, "messages.csv"))
        self.images_df = pd.read_csv(os.path.join(self.data_dir, "images.csv"))
        
        self._resolve_missing_event_amounts()
        return self

    def _resolve_missing_event_amounts(self):
        """
        Locates rows with missing amounts, joins with images.csv
        via event_id == related_event_id, and extracts amounts.
        """
        missing_mask = self.events_df["amount"].isna()
        if not missing_mask.any():
            return

        # Merge missing events with images.csv to locate relevant image files
        missing_events = self.events_df[missing_mask]
        merged = missing_events.merge(
            self.images_df,
            left_on="event_id",
            right_on="related_event_id",
            how="inner"
        )

        for _, row in merged.iterrows():
            event_id = row["event_id"]
            image_filename = f"{row['image_id']}.png"
            image_path = os.path.join(MEDIA_DIR, image_filename)
            
            extracted_val = self._extract_amount_from_image(image_path)
            if extracted_val is not None:
                self.events_df.loc[self.events_df["event_id"] == event_id, "amount"] = extracted_val

    def _extract_amount_from_image(self, image_path: str) -> Optional[float]:
        """
        Placeholder for OCR / Vision-based amount extraction.
        Extracts numerical expense value from receipt/invoice PNGs.
        """
        if not os.path.exists(image_path):
            return None
        # Hook OCR (e.g. pytesseract, easyocr, or vision API call) here
        return None

    def convert_to_home_currency(self, amount: float, from_curr: str, to_curr: str, date_str: str) -> float:
        """
        Converts foreign currency amounts to user's home_currency 
        using exchange_rates.csv.
        """
        if from_curr == to_curr or pd.isna(from_curr):
            return float(amount)

        rate_match = self.rates_df[
            (self.rates_df["date"] == date_str) &
            (self.rates_df["from_currency"] == from_curr) &
            (self.rates_df["to_currency"] == to_curr)
        ]

        if not rate_match.empty:
            rate = float(rate_match.iloc[0]["rate"])
            return float(amount) * rate

        # Handle inverse rate lookup if direct pair is missing
        inv_match = self.rates_df[
            (self.rates_df["date"] == date_str) &
            (self.rates_df["from_currency"] == to_curr) &
            (self.rates_df["to_currency"] == from_curr)
        ]
        if not inv_match.empty:
            rate = float(inv_match.iloc[0]["rate"])
            return float(amount) / rate

        return float(amount)

    def get_user_context(self, user_id: str, request_id: str) -> Dict[str, Any]:
        """Bundles all relevant tabular, message, and option data for a single request."""
        profile = self.profiles_df[self.profiles_df["user_id"] == user_id].iloc[0].to_dict()
        events = self.events_df[self.events_df["user_id"] == user_id].copy()
        req = self.requests_df[self.requests_df["request_id"] == request_id].iloc[0].to_dict()
        
        user_messages = self.messages_df[
            (self.messages_df["user_id"] == user_id) | 
            (self.messages_df["request_id"] == request_id)
        ]
        options = self.options_df[self.options_df["request_id"] == request_id].to_dict(orient="records")

        return {
            "profile": profile,
            "request": req,
            "events": events,
            "messages": user_messages,
            "payment_options": options
        }
