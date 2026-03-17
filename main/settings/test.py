import typed_settings as ts
import dataclasses


@dataclasses.dataclass(frozen=True)
class Settings:
    username: str
    password: ts.SecretStr


settings = Settings("monty", ts.SecretStr("S3cr3t!"))
assert str(settings) == "Settings(username='monty', password='*******')"
