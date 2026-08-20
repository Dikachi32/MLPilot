"""Tests for subscription and payment services."""
import pytest
from services.payment_service import DemoPaymentService
from services.subscription_service import SubscriptionService

def test_create_checkout():
    session = DemoPaymentService.create_checkout("pro_monthly", "monthly", "test@example.com")
    assert session["status"] == "pending"
    assert session["demo_warning"] == "DEMO PAYMENT — NO REAL TRANSACTION WILL OCCUR"
    assert session["amount"] == 14.99

def test_process_payment():
    result = DemoPaymentService.process_payment("demo_sess_123", "pro_weekly", "weekly")
    assert result["status"] == "paid"
    assert result["demo_warning"] == "DEMO PAYMENT — NO REAL TRANSACTION OCCURRED"

def test_verify_session():
    assert DemoPaymentService.verify_session("demo_sess_abc123") == True
    assert DemoPaymentService.verify_session("invalid") == False