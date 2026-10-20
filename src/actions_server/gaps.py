"""Functions to interact with the corpus_gaps table in DynamoDB."""

import os
import hashlib
from datetime import datetime, timezone
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
    return get_dynamodb_resource().Table(TABLE)

def normalize(question):
    return " ".join(question.split()).lower()


def log_gap(table, question, urls):
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