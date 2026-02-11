import os


def get_auth_secret_env():
    auth_secret = os.getenv("AUTH_SECRET")
    assert auth_secret, "AUTH_SECRET is not set"
    return auth_secret


def get_debug_token_env():
    debug_token = os.getenv("DEBUG_TOKEN")
    assert debug_token, "DEBUG_TOKEN is not set"
    return debug_token


def get_debug_env():
    return os.getenv("DEBUG")

def get_jobs_worker_env():
    return os.getenv("JOBS_WORKER")

def get_aws_access_key_id_env():
    aws_access_key_id = os.getenv("PRESENTON_AWS_ACCESS_KEY_ID")
    return aws_access_key_id


def get_aws_secret_access_key_env():
    aws_secret_access_key = os.getenv("PRESENTON_AWS_SECRET_ACCESS_KEY")
    return aws_secret_access_key


def get_aws_region_env():
    region = os.getenv("PRESENTON_AWS_REGION")
    assert region, "PRESENTON_AWS_REGION is not set"
    return region


def get_s3_public_bucket_env():
    bucket = os.getenv("S3_PUBLIC_BUCKET")
    assert bucket, "S3_PUBLIC_BUCKET is not set"
    return bucket


def get_s3_private_bucket_env():
    bucket = os.getenv("S3_PRIVATE_BUCKET")
    assert bucket, "S3_PRIVATE_BUCKET is not set"
    return bucket


def get_s3_temporary_bucket_env():
    bucket = os.getenv("S3_TEMPORARY_BUCKET")
    assert bucket, "S3_TEMPORARY_BUCKET is not set"
    return bucket


def get_cloudwatch_log_group_env():
    assert os.getenv("CLOUDWATCH_LOG_GROUP"), "CLOUDWATCH_LOG_GROUP is not set"
    return os.getenv("CLOUDWATCH_LOG_GROUP")


def get_export_url_env():
    assert os.getenv("EXPORT_URL", "EXPORT_URL is not set")
    return os.getenv("EXPORT_URL")


def get_templates_schema_url_env():
    url = os.getenv("TEMPLATES_SCHEMA_URL")
    assert url, "TEMPLATES_SCHEMA_URL is not set"
    return url


def get_database_url_env():
    return os.getenv("DATABASE_URL")


def get_database_host_env():
    return os.getenv("DATABASE_HOST")


def get_database_port_env():
    return os.getenv("DATABASE_PORT")


def get_database_user_env():
    return os.getenv("DATABASE_USER")


def get_database_pass_env():
    return os.getenv("DATABASE_PASS")


def get_database_name_env():
    return os.getenv("DATABASE_NAME")


def get_database_args_env():
    return os.getenv("DATABASE_ARGS")


def get_temp_directory_env():
    temp_dir = os.getenv("TEMP_DIRECTORY")
    assert temp_dir, "TEMP_DIRECTORY is not set"
    return temp_dir


def get_llm_provider_env():
    llm_provider = os.getenv("LLM")
    assert llm_provider, "LLM is not set"
    return llm_provider


def get_outlines_llm_provider_env():
    return os.getenv("OUTLINES_LLM")


def get_anthropic_api_key_env():
    return os.getenv("ANTHROPIC_API_KEY")


def get_anthropic_model_env():
    return os.getenv("ANTHROPIC_MODEL")


def get_ollama_url_env():
    return os.getenv("OLLAMA_URL")


def get_custom_llm_url_env():
    return os.getenv("CUSTOM_LLM_URL")


def get_openai_api_key_env():
    return os.getenv("OPENAI_API_KEY")


def get_openai_model_env():
    return os.getenv("OPENAI_MODEL")


def get_google_api_key_env():
    return os.getenv("GOOGLE_API_KEY")


def get_google_model_env():
    return os.getenv("GOOGLE_MODEL")


def get_custom_llm_api_key_env():
    return os.getenv("CUSTOM_LLM_API_KEY")


def get_ollama_model_env():
    return os.getenv("OLLAMA_MODEL")


def get_custom_model_env():
    return os.getenv("CUSTOM_MODEL")


def get_cerebras_api_key_env():
    return os.getenv("CEREBRAS_API_KEY")


def get_openrouter_api_key_env():
    return os.getenv("OPENROUTER_API_KEY")


def get_zai_api_key_env():
    return os.getenv("ZAI_API_KEY")


def get_exa_api_key_env():
    return os.getenv("EXA_API_KEY")


def get_tavily_api_key_env():
    return os.getenv("TAVILY_API_KEY")


def get_image_provider_env():
    return os.getenv("IMAGE_PROVIDER")


def get_pexels_api_key_env():
    return os.getenv("PEXELS_API_KEY")


def get_pixabay_api_key_env():
    return os.getenv("PIXABAY_API_KEY")


def get_replicate_api_key_env():
    return os.getenv("REPLICATE_API_KEY")


def get_tool_calls_env():
    return os.getenv("TOOL_CALLS")


def get_disable_thinking_env():
    return os.getenv("DISABLE_THINKING")


def get_extended_reasoning_env():
    return os.getenv("EXTENDED_REASONING")


def get_web_grounding_env():
    return os.getenv("WEB_GROUNDING")


# Stripe
def get_stripe_secret_key_env():
    return os.getenv("STRIPE_SECRET_KEY")


def get_stripe_webhook_secret_env():
    return os.getenv("STRIPE_WEBHOOK_SECRET")


# Resend
def get_resend_api_key_env():
    return os.getenv("RESEND_API_KEY")


# Frontend
def get_nextjs_url_env():
    url = os.getenv("NEXTJS_URL")
    assert url, "NEXTJS_URL is not set"
    return url


# Google
def get_google_client_id_env():
    return os.getenv("GOOGLE_CLIENT_ID")


def get_google_client_secret_env():
    return os.getenv("GOOGLE_CLIENT_SECRET")


# Cloudflare
def get_cloudflare_account_id_env():
    return os.getenv("CLOUDFLARE_ACCOUNT_ID")


def get_cloudflare_api_key_env():
    return os.getenv("CLOUDFLARE_API_KEY")


def get_cloudflare_image_generation_model_env():
    return os.getenv("CLOUDFLARE_IMAGE_GENERATION_MODEL")


def get_libreoffice_path_env():
    return os.getenv("LIBREOFFICE_PATH")


# Others
def get_disable_credits_env():
    return os.getenv("DISABLE_CREDITS")
