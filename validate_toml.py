import tomli # type: ignore

with open("pyproject.toml", "rb") as f:
    try:
        data = tomli.load(f)
        print("✅ TOML parsed successfully.")
    except tomli.TOMLDecodeError as e:
        print(f"❌ TOML parsing error: {e}")
