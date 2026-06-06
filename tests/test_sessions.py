"""Sessions / messages SQLite layer."""

from __future__ import annotations


def test_create_and_get(isolated_paths):
    from personal_db import sessions as s
    sess = s.create_session()
    assert sess.id and len(sess.id) >= 8
    assert sess.title == "New chat"
    assert sess.summary == ""
    assert sess.summary_up_to_msg_id == 0
    got = s.get_session(sess.id)
    assert got is not None
    assert got.id == sess.id


def test_list_orders_by_updated(isolated_paths):
    from personal_db import sessions as s
    a = s.create_session(title="A")
    b = s.create_session(title="B")
    # touch a so it bubbles to top
    s.add_message(a.id, "user", "hello")
    rows = s.list_sessions()
    assert rows[0].id == a.id
    assert rows[1].id == b.id


def test_add_message_returns_id_and_orders(isolated_paths):
    from personal_db import sessions as s
    sess = s.create_session()
    m1 = s.add_message(sess.id, "user", "first")
    m2 = s.add_message(sess.id, "assistant", "second")
    m3 = s.add_message(sess.id, "user", "third")
    assert m1 < m2 < m3
    msgs = s.get_messages(sess.id)
    assert [m.content for m in msgs] == ["first", "second", "third"]
    assert msgs[0].id == m1


def test_get_messages_since_id(isolated_paths):
    from personal_db import sessions as s
    sess = s.create_session()
    a = s.add_message(sess.id, "user", "a")
    b = s.add_message(sess.id, "assistant", "b")
    c = s.add_message(sess.id, "user", "c")
    later = s.get_messages(sess.id, since_id=a)
    assert [m.id for m in later] == [b, c]


def test_update_summary_persists(isolated_paths):
    from personal_db import sessions as s
    sess = s.create_session()
    mid = s.add_message(sess.id, "user", "hello")
    s.update_summary(sess.id, "the summary", mid)
    fresh = s.get_session(sess.id)
    assert fresh.summary == "the summary"
    assert fresh.summary_up_to_msg_id == mid


def test_rename_session_updates_title_and_timestamp(isolated_paths):
    from personal_db import sessions as s
    sess = s.create_session()
    old = sess.updated_at
    s.rename_session(sess.id, "new title")
    fresh = s.get_session(sess.id)
    assert fresh.title == "new title"
    assert fresh.updated_at >= old


def test_delete_cascades_messages(isolated_paths):
    from personal_db import sessions as s
    sess = s.create_session()
    s.add_message(sess.id, "user", "x")
    s.add_message(sess.id, "assistant", "y")
    s.delete_session(sess.id)
    assert s.get_session(sess.id) is None
    assert s.get_messages(sess.id) == []
