
def test_parser():
    from src.data.parser import template_of
    templ, toks = template_of("GET /api/v1/users/123 200 OK svc=user")
    assert "<*>" in templ
    assert len(toks) >= 4
