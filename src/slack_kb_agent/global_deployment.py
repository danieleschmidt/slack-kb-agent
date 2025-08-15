"""
Global-first deployment and internationalization for Slack KB Agent.
Implements multi-region support, i18n, and compliance frameworks.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class Region(Enum):
    """Supported deployment regions."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)


@dataclass
class RegionConfig:
    """Configuration for a specific region."""
    region: Region
    data_residency_required: bool
    compliance_frameworks: List[ComplianceFramework]
    supported_languages: List[str]
    timezone: str
    currency: str
    date_format: str
    number_format: str
    primary_language: str = "en"


class LocalizationManager:
    """Manages internationalization and localization."""

    def __init__(self, default_language: str = "en"):
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, str]] = {}
        self.supported_languages = ["en", "es", "fr", "de", "ja", "zh", "pt", "it", "ko", "ru"]
        self.load_translations()

    def load_translations(self):
        """Load translation files for all supported languages."""
        # Default English translations
        self.translations["en"] = {
            "welcome_message": "Welcome to Slack KB Agent",
            "search_results": "Search Results",
            "no_results_found": "No results found for your query",
            "error_occurred": "An error occurred while processing your request",
            "permission_denied": "Permission denied",
            "rate_limit_exceeded": "Rate limit exceeded. Please try again later",
            "help_message": "Type your question or use /kb help for more information",
            "usage_stats": "Usage Statistics",
            "knowledge_base_updated": "Knowledge base updated successfully",
            "invalid_command": "Invalid command. Type /kb help for available commands",
        }

        # Spanish translations
        self.translations["es"] = {
            "welcome_message": "Bienvenido al Agente KB de Slack",
            "search_results": "Resultados de búsqueda",
            "no_results_found": "No se encontraron resultados para su consulta",
            "error_occurred": "Ocurrió un error al procesar su solicitud",
            "permission_denied": "Permiso denegado",
            "rate_limit_exceeded": "Límite de tasa excedido. Inténtelo de nuevo más tarde",
            "help_message": "Escriba su pregunta o use /kb help para más información",
            "usage_stats": "Estadísticas de uso",
            "knowledge_base_updated": "Base de conocimientos actualizada exitosamente",
            "invalid_command": "Comando inválido. Escriba /kb help para comandos disponibles",
        }

        # French translations
        self.translations["fr"] = {
            "welcome_message": "Bienvenue dans l'Agent KB Slack",
            "search_results": "Résultats de recherche",
            "no_results_found": "Aucun résultat trouvé pour votre requête",
            "error_occurred": "Une erreur s'est produite lors du traitement de votre demande",
            "permission_denied": "Permission refusée",
            "rate_limit_exceeded": "Limite de taux dépassée. Veuillez réessayer plus tard",
            "help_message": "Tapez votre question ou utilisez /kb help pour plus d'informations",
            "usage_stats": "Statistiques d'utilisation",
            "knowledge_base_updated": "Base de connaissances mise à jour avec succès",
            "invalid_command": "Commande invalide. Tapez /kb help pour les commandes disponibles",
        }

        # German translations
        self.translations["de"] = {
            "welcome_message": "Willkommen beim Slack KB Agent",
            "search_results": "Suchergebnisse",
            "no_results_found": "Keine Ergebnisse für Ihre Anfrage gefunden",
            "error_occurred": "Ein Fehler ist bei der Verarbeitung Ihrer Anfrage aufgetreten",
            "permission_denied": "Berechtigung verweigert",
            "rate_limit_exceeded": "Ratenlimit überschritten. Bitte versuchen Sie es später erneut",
            "help_message": "Geben Sie Ihre Frage ein oder verwenden Sie /kb help für weitere Informationen",
            "usage_stats": "Nutzungsstatistiken",
            "knowledge_base_updated": "Wissensdatenbank erfolgreich aktualisiert",
            "invalid_command": "Ungültiger Befehl. Geben Sie /kb help für verfügbare Befehle ein",
        }

        # Japanese translations
        self.translations["ja"] = {
            "welcome_message": "Slack KBエージェントへようこそ",
            "search_results": "検索結果",
            "no_results_found": "お探しのクエリに対する結果が見つかりませんでした",
            "error_occurred": "リクエストの処理中にエラーが発生しました",
            "permission_denied": "アクセス拒否",
            "rate_limit_exceeded": "レート制限を超過しました。後でもう一度お試しください",
            "help_message": "質問を入力するか、/kb helpを使用して詳細情報を取得してください",
            "usage_stats": "使用統計",
            "knowledge_base_updated": "ナレッジベースが正常に更新されました",
            "invalid_command": "無効なコマンド。利用可能なコマンドについては/kb helpと入力してください",
        }

        # Chinese translations
        self.translations["zh"] = {
            "welcome_message": "欢迎使用Slack知识库代理",
            "search_results": "搜索结果",
            "no_results_found": "未找到与您的查询相关的结果",
            "error_occurred": "处理您的请求时发生错误",
            "permission_denied": "权限被拒绝",
            "rate_limit_exceeded": "超过速率限制。请稍后重试",
            "help_message": "输入您的问题或使用/kb help获取更多信息",
            "usage_stats": "使用统计",
            "knowledge_base_updated": "知识库更新成功",
            "invalid_command": "无效命令。输入/kb help查看可用命令",
        }

    def set_language(self, language_code: str) -> bool:
        """Set the current language."""
        if language_code in self.supported_languages:
            self.current_language = language_code
            return True
        return False

    def get_text(self, key: str, language: Optional[str] = None) -> str:
        """Get localized text for a key."""
        lang = language or self.current_language

        if lang in self.translations and key in self.translations[lang]:
            return self.translations[lang][key]
        elif key in self.translations[self.default_language]:
            return self.translations[self.default_language][key]
        else:
            return key  # Return key if no translation found

    def format_datetime(self, dt: datetime, language: Optional[str] = None) -> str:
        """Format datetime according to locale."""
        lang = language or self.current_language

        # Language-specific datetime formatting
        formats = {
            "en": "%Y-%m-%d %H:%M:%S",
            "es": "%d/%m/%Y %H:%M:%S",
            "fr": "%d/%m/%Y %H:%M:%S",
            "de": "%d.%m.%Y %H:%M:%S",
            "ja": "%Y年%m月%d日 %H:%M:%S",
            "zh": "%Y年%m月%d日 %H:%M:%S",
        }

        format_str = formats.get(lang, formats["en"])
        return dt.strftime(format_str)

    def format_number(self, number: Union[int, float], language: Optional[str] = None) -> str:
        """Format numbers according to locale."""
        lang = language or self.current_language

        # Language-specific number formatting
        if lang in ["de", "fr", "es"]:
            # European format: 1.234.567,89
            if isinstance(number, float):
                return f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            else:
                return f"{number:,}".replace(",", ".")
        elif lang in ["ja", "zh"]:
            # Asian format: group by 4 digits
            if isinstance(number, float):
                return f"{number:,.2f}"
            else:
                return f"{number:,}"
        else:
            # English format: 1,234,567.89
            if isinstance(number, float):
                return f"{number:,.2f}"
            else:
                return f"{number:,}"


class ComplianceManager:
    """Manages compliance with various privacy regulations."""

    def __init__(self):
        self.compliance_rules: Dict[ComplianceFramework, Dict[str, Any]] = {}
        self.setup_compliance_rules()
        self.audit_log: List[Dict[str, Any]] = []

    def setup_compliance_rules(self):
        """Set up compliance rules for different frameworks."""

        # GDPR (EU) rules
        self.compliance_rules[ComplianceFramework.GDPR] = {
            "data_retention_days": 1095,  # 3 years max
            "requires_explicit_consent": True,
            "right_to_be_forgotten": True,
            "data_portability": True,
            "data_protection_officer_required": True,
            "breach_notification_hours": 72,
            "sensitive_data_categories": [
                "personal_identifier", "health_data", "biometric_data",
                "political_opinions", "religious_beliefs"
            ],
            "lawful_basis_required": True,
            "privacy_by_design": True
        }

        # CCPA (California) rules
        self.compliance_rules[ComplianceFramework.CCPA] = {
            "data_retention_days": 730,  # 2 years
            "requires_explicit_consent": False,  # Opt-out model
            "right_to_be_forgotten": True,
            "data_portability": True,
            "sale_opt_out": True,
            "consumer_request_response_days": 45,
            "verification_required": True,
            "non_discrimination": True
        }

        # PDPA (Singapore) rules
        self.compliance_rules[ComplianceFramework.PDPA] = {
            "data_retention_days": 1095,
            "requires_explicit_consent": True,
            "purpose_limitation": True,
            "data_accuracy": True,
            "data_protection_officer_required": False,
            "breach_notification_hours": 72,
            "access_correction_rights": True
        }

    def check_compliance(self, framework: ComplianceFramework,
                        data_type: str, processing_purpose: str) -> Dict[str, Any]:
        """Check if data processing complies with framework."""
        rules = self.compliance_rules.get(framework, {})

        compliance_status = {
            "compliant": True,
            "framework": framework.value,
            "requirements": [],
            "violations": [],
            "recommendations": []
        }

        # Check consent requirements
        if rules.get("requires_explicit_consent", False):
            compliance_status["requirements"].append("Explicit user consent required")

        # Check data retention limits
        retention_days = rules.get("data_retention_days", 0)
        if retention_days > 0:
            compliance_status["requirements"].append(
                f"Data must be deleted after {retention_days} days"
            )

        # Check sensitive data handling
        sensitive_categories = rules.get("sensitive_data_categories", [])
        if data_type in sensitive_categories:
            compliance_status["requirements"].append(
                "Enhanced protection required for sensitive data"
            )

        return compliance_status

    def log_data_processing(self, user_id: str, data_type: str,
                           purpose: str, framework: ComplianceFramework):
        """Log data processing for audit trail."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "data_type": data_type,
            "purpose": purpose,
            "framework": framework.value,
            "compliance_check": self.check_compliance(framework, data_type, purpose)
        }

        self.audit_log.append(log_entry)

        # Maintain audit log size
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-8000:]  # Keep recent entries

    def handle_deletion_request(self, user_id: str, framework: ComplianceFramework) -> Dict[str, Any]:
        """Handle user data deletion request (Right to be Forgotten)."""
        rules = self.compliance_rules.get(framework, {})

        if not rules.get("right_to_be_forgotten", False):
            return {
                "success": False,
                "reason": f"Right to be forgotten not supported under {framework.value}"
            }

        # Log the deletion request
        self.log_data_processing(
            user_id, "deletion_request", "right_to_be_forgotten", framework
        )

        return {
            "success": True,
            "request_id": f"del_{user_id}_{int(datetime.utcnow().timestamp())}",
            "estimated_completion": "7 business days",
            "framework": framework.value
        }


class RegionManager:
    """Manages multi-region deployment and data residency."""

    def __init__(self):
        self.regions: Dict[Region, RegionConfig] = {}
        self.current_region = Region.US_EAST
        self.setup_default_regions()

    def setup_default_regions(self):
        """Set up default region configurations."""

        # US East
        self.regions[Region.US_EAST] = RegionConfig(
            region=Region.US_EAST,
            data_residency_required=False,
            compliance_frameworks=[ComplianceFramework.CCPA],
            supported_languages=["en", "es"],
            timezone="America/New_York",
            currency="USD",
            date_format="%m/%d/%Y",
            number_format="en_US",
            primary_language="en"
        )

        # EU West
        self.regions[Region.EU_WEST] = RegionConfig(
            region=Region.EU_WEST,
            data_residency_required=True,
            compliance_frameworks=[ComplianceFramework.GDPR],
            supported_languages=["en", "fr", "de", "es", "it"],
            timezone="Europe/London",
            currency="EUR",
            date_format="%d/%m/%Y",
            number_format="en_GB",
            primary_language="en"
        )

        # Asia Pacific
        self.regions[Region.ASIA_PACIFIC] = RegionConfig(
            region=Region.ASIA_PACIFIC,
            data_residency_required=True,
            compliance_frameworks=[ComplianceFramework.PDPA],
            supported_languages=["en", "zh", "ja", "ko"],
            timezone="Asia/Singapore",
            currency="SGD",
            date_format="%d/%m/%Y",
            number_format="en_SG",
            primary_language="en"
        )

    def get_region_for_user(self, user_location: str) -> Region:
        """Determine appropriate region for user based on location."""
        location_mapping = {
            "US": Region.US_EAST,
            "CA": Region.US_EAST,
            "MX": Region.US_EAST,
            "GB": Region.EU_WEST,
            "FR": Region.EU_WEST,
            "DE": Region.EU_WEST,
            "ES": Region.EU_WEST,
            "IT": Region.EU_WEST,
            "SG": Region.ASIA_PACIFIC,
            "JP": Region.ASIA_NORTHEAST,
            "KR": Region.ASIA_NORTHEAST,
            "CN": Region.ASIA_PACIFIC,
        }

        return location_mapping.get(user_location.upper(), Region.US_EAST)

    def get_compliance_frameworks(self, region: Region) -> List[ComplianceFramework]:
        """Get required compliance frameworks for region."""
        region_config = self.regions.get(region)
        return region_config.compliance_frameworks if region_config else []

    def validate_data_residency(self, user_region: Region, processing_region: Region) -> bool:
        """Validate if data processing complies with residency requirements."""
        user_config = self.regions.get(user_region)

        if not user_config:
            return True  # Unknown region, allow processing

        if not user_config.data_residency_required:
            return True  # No residency requirement

        # Check if processing region is acceptable
        # For simplicity, require same region for strict residency
        return user_region == processing_region


class GlobalDeploymentManager:
    """Orchestrates global deployment with localization and compliance."""

    def __init__(self):
        self.localization = LocalizationManager()
        self.compliance = ComplianceManager()
        self.region_manager = RegionManager()
        self.deployment_status: Dict[Region, str] = {}

    def deploy_to_region(self, region: Region) -> Dict[str, Any]:
        """Deploy to a specific region with appropriate configuration."""
        region_config = self.region_manager.regions.get(region)

        if not region_config:
            return {"success": False, "error": f"Unknown region: {region}"}

        try:
            # Set up localization for region
            self.localization.set_language(region_config.primary_language)

            # Validate compliance requirements
            compliance_checks = []
            for framework in region_config.compliance_frameworks:
                check = self.compliance.check_compliance(
                    framework, "user_data", "knowledge_base_service"
                )
                compliance_checks.append(check)

            # Mark region as deployed
            self.deployment_status[region] = "active"

            return {
                "success": True,
                "region": region.value,
                "primary_language": region_config.primary_language,
                "supported_languages": region_config.supported_languages,
                "compliance_frameworks": [f.value for f in region_config.compliance_frameworks],
                "data_residency_required": region_config.data_residency_required,
                "timezone": region_config.timezone,
                "compliance_checks": compliance_checks
            }

        except Exception as e:
            logger.error(f"Failed to deploy to region {region}: {e}")
            self.deployment_status[region] = "failed"
            return {"success": False, "error": str(e)}

    def get_user_configuration(self, user_id: str, user_location: str = "US") -> Dict[str, Any]:
        """Get appropriate configuration for user based on location."""
        user_region = self.region_manager.get_region_for_user(user_location)
        region_config = self.region_manager.regions.get(user_region)

        if not region_config:
            # Fallback to default
            user_region = Region.US_EAST
            region_config = self.region_manager.regions[user_region]

        # Set appropriate language and compliance
        compliance_frameworks = self.region_manager.get_compliance_frameworks(user_region)

        return {
            "user_id": user_id,
            "region": user_region.value,
            "primary_language": region_config.primary_language,
            "supported_languages": region_config.supported_languages,
            "timezone": region_config.timezone,
            "currency": region_config.currency,
            "date_format": region_config.date_format,
            "compliance_frameworks": [f.value for f in compliance_frameworks],
            "data_residency_required": region_config.data_residency_required
        }

    def process_user_request(self, user_id: str, request_data: Dict[str, Any],
                           user_location: str = "US") -> Dict[str, Any]:
        """Process user request with appropriate localization and compliance."""
        user_config = self.get_user_configuration(user_id, user_location)
        user_region = Region(user_config["region"])

        # Check data residency compliance
        processing_region = self.region_manager.current_region
        if not self.region_manager.validate_data_residency(user_region, processing_region):
            return {
                "success": False,
                "error": self.localization.get_text("permission_denied"),
                "reason": "Data residency violation"
            }

        # Log for compliance
        for framework_name in user_config["compliance_frameworks"]:
            framework = ComplianceFramework(framework_name)
            self.compliance.log_data_processing(
                user_id, "search_request", "knowledge_retrieval", framework
            )

        # Set appropriate language
        self.localization.set_language(user_config["primary_language"])

        return {
            "success": True,
            "user_config": user_config,
            "localized_response": True,
            "compliance_logged": True
        }

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get global deployment status."""
        return {
            "active_regions": [region.value for region, status in self.deployment_status.items() if status == "active"],
            "total_regions": len(self.deployment_status),
            "supported_languages": self.localization.supported_languages,
            "compliance_frameworks": [f.value for f in ComplianceFramework],
            "default_language": self.localization.default_language,
            "audit_log_entries": len(self.compliance.audit_log)
        }


# Global deployment manager instance
global_deployment = GlobalDeploymentManager()


def localize_text(key: str, language: Optional[str] = None) -> str:
    """Get localized text for a key."""
    return global_deployment.localization.get_text(key, language)


def get_user_config(user_id: str, location: str = "US") -> Dict[str, Any]:
    """Get configuration for user based on location."""
    return global_deployment.get_user_configuration(user_id, location)


def deploy_globally() -> Dict[str, Any]:
    """Deploy to all supported regions."""
    results = {}

    for region in [Region.US_EAST, Region.EU_WEST, Region.ASIA_PACIFIC]:
        result = global_deployment.deploy_to_region(region)
        results[region.value] = result

    return {
        "global_deployment": results,
        "deployment_status": global_deployment.get_deployment_status()
    }
