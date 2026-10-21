"""Functions to interact with the corpus_gaps table in DynamoDB."""

import os
import hashlib
from datetime import datetime, timezone
from urllib.parse import urlparse
from functools import cache

import boto3
from src.constants import Constants

@cache
def get_dynamodb_resource():
    kwargs = {}
    if Constants.DYNAMODB_ENDPOINT:
        # DynamoDB Local ignores credentials, but boto3 insists on having some.
        kwargs = {
            "endpoint_url": Constants.DYNAMODB_ENDPOINT,
            "aws_access_key_id": "local",
            "aws_secret_access_key": "local",
        }
    return boto3.resource("dynamodb", region_name=Constants.AWS_REGION, **kwargs)

@cache
def get_table():
    return get_dynamodb_resource().Table(Constants.GAPS_TABLE)

def normalize(question):
    return " ".join(question.split()).lower()

def log_gap(table, question, urls):
    validate(question, urls)
    key = hashlib.sha256(normalize(question).encode()).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    response = table.update_item(
        Key={"question_hash": key},
        UpdateExpression=(
            "ADD #count :one "
            "SET #q = if_not_exists(#q, :q), "
            "#urls = :urls, "
            "#first = if_not_exists(#first, :now), "
            "#last = :now"
        ),
        ExpressionAttributeNames={
            "#count": "count",
            "#q": "question",
            "#urls": "urls",
            "#first": "first_seen",
            "#last": "last_seen",
        },
        ExpressionAttributeValues={":one": 1, ":q": question, ":urls": urls, ":now": now},
        ReturnValues="UPDATED_NEW",
    )
    return {"question_hash": key, "count": int(response["Attributes"]["count"])}

def validate(question, urls):
    if not isinstance(question, str):
        raise ValueError("Question must be a string.")
    if len(question) > Constants.MAX_QUESTION_LEN:
        raise ValueError(f"Question exceeds maximum length of {Constants.MAX_QUESTION_LEN} characters.")
    question_normalized = normalize(question)
    if not question_normalized:
        raise ValueError("Question cannot be empty or whitespace.")
    if not isinstance(urls, list) or len(urls) > Constants.MAX_URLS_PER_GAP:
        raise ValueError(f"URLs must be a list of strings less than {Constants.MAX_URLS_PER_GAP}.")
    for url in urls:
        if not isinstance(url, str) or len(url) > Constants.MAX_URL_LEN:
            raise ValueError(f"Each URL must be a string less than {Constants.MAX_URL_LEN} characters.")
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"Not a valid http(s) URL: {url}")