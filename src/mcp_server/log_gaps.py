"""Scratch script: round trip against DynamoDB Local for the corpus_gaps table."""

import hashlib
from datetime import datetime, timezone

import boto3

TABLE = "corpus_gaps"

dynamodb = boto3.resource(
    "dynamodb",
    endpoint_url="http://127.0.0.1:8001",
    region_name="ap-south-1",
    aws_access_key_id="local",       # DynamoDB Local ignores these, boto3 wants them
    aws_secret_access_key="local",
)


def make_table():
    if TABLE in [t.name for t in dynamodb.tables.all()]:
        return dynamodb.Table(TABLE)
    table = dynamodb.create_table(
        TableName=TABLE,
        KeySchema=[{"AttributeName": "question_hash", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "question_hash", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    table.wait_until_exists()
    return table


def log_gap(table, question, urls):
    key = hashlib.sha256(question.strip().lower().encode()).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    table.update_item(
        Key={"question_hash": key},
        # Names are aliased (#x) so a reserved word like "count" can't bite us.
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
    )
    return key


if __name__ == "__main__":
    table = make_table()
    urls = ["https://example.org/source-1"]

    key = log_gap(table, "Early signs of autism in toddlers", urls)
    print("after 1st:", table.get_item(Key={"question_hash": key})["Item"])

    log_gap(table, "  early signs of AUTISM in toddlers ", urls)  # same after normalizing
    print("after 2nd:", table.get_item(Key={"question_hash": key})["Item"])

    log_gap(table, "What are early signs of autism in toddlers", urls)  # paraphrase
    for item in table.scan()["Items"]:
        print(item["count"], item["question"])