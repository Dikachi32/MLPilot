"""Demo payment processing — NO REAL TRANSACTIONS."""
import uuid
from datetime import datetime
from typing import Dict

class DemoPaymentService:
    DEMO_GATEWAY = "DEMO_GATEWAY"

    @staticmethod
    def create_checkout(plan: str, billing_cycle: str, user_email: str) -> Dict:
        """Create a demo checkout session."""
        session_id = f"demo_sess_{uuid.uuid4().hex[:12]}"

        pricing = {
            "pro_weekly": {"weekly": 4.99, "monthly": None},
            "pro_monthly": {"weekly": None, "monthly": 14.99},
        }

        amount = pricing.get(plan, {}).get(billing_cycle, 0.0)

        return {
            "session_id": session_id,
            "plan": plan,
            "billing_cycle": billing_cycle,
            "amount": amount,
            "currency": "USD",
            "status": "pending",
            "gateway": DemoPaymentService.DEMO_GATEWAY,
            "demo_warning": "DEMO PAYMENT — NO REAL TRANSACTION WILL OCCUR",
            "created_at": datetime.utcnow().isoformat(),
        }

    @staticmethod
    def process_payment(session_id: str, plan: str, billing_cycle: str) -> Dict:
        """Simulate payment processing."""
        return {
            "session_id": session_id,
            "plan": plan,
            "billing_cycle": billing_cycle,
            "status": "paid",
            "gateway": DemoPaymentService.DEMO_GATEWAY,
            "transaction_id": f"demo_txn_{uuid.uuid4().hex[:16]}",
            "paid_at": datetime.utcnow().isoformat(),
            "demo_warning": "DEMO PAYMENT — NO REAL TRANSACTION OCCURRED",
        }

    @staticmethod
    def verify_session(session_id: str) -> bool:
        return session_id.startswith("demo_sess_")