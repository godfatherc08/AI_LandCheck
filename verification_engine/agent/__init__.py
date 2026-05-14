"""Verification Agent Module - Offline document verification and report generation"""

from verification_engine.agent.verifier import verify_document_offline
from verification_engine.agent.verification_report import generate_verification_report

__all__ = ['verify_document_offline', 'generate_verification_report']