from pathlib import Path

conf_template = Path(__file__).with_name("conf.py.in")
namespace = {
    "__file__": str(conf_template),
    "__name__": "__main__",
}

rendered = conf_template.read_text(encoding="utf-8").replace(
    "@PACKAGE_VERSION@",
    "dev",
)

exec(compile(rendered, str(conf_template), "exec"), namespace)

globals().update(namespace)
