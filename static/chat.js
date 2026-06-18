(async function () {
  const chatEl = document.getElementById("chat");
  const msgEl = document.getElementById("message");
  const sendBtn = document.getElementById("send");

  function append(text, cls) {
    const d = document.createElement("div");
    d.className = "msg " + cls;
    d.textContent = text;
    chatEl.appendChild(d);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  function getSessionId() {
    return localStorage.getItem("hc_session_id") || null;
  }
  function setSessionId(id) {
    localStorage.setItem("hc_session_id", id);
  }

  async function sendMessage() {
    const text = msgEl.value.trim();
    if (!text) return;
    append(text, "user");
    msgEl.value = "";
    const payload = { message: text };
    const sid = getSessionId();
    if (sid) payload.session_id = sid;

    append("...", "assistant");
    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      // remove the "..." last assistant message
      const last = Array.from(chatEl.getElementsByClassName("assistant")).pop();
      if (last) last.remove();

      if (data.reply) {
        append(data.reply, "assistant");
      } else {
        append("No reply from server.", "assistant");
      }
      if (data.session_id) setSessionId(data.session_id);
    } catch (err) {
      append("Error contacting server.", "assistant");
    }
  }

  sendBtn.addEventListener("click", sendMessage);
  msgEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
})();
