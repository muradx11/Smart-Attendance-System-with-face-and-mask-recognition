const cameraFeed = document.getElementById("cameraFeed");
const totalElement = document.getElementById("totalStudents");
let totalStudents = Number(
    totalElement.dataset.total
);
let presentStudents = 0;
let sessionRunning = false;
let timeLeft = 10 * 60;

let timerInterval = null;
let statusPollInterval = null;
let livePollInterval = null;


const startBtn = document.getElementById("startBtn");
const endBtn = document.getElementById("endBtn");
const exportBtn = document.getElementById("exportBtn");
const exportSessionBtn = document.getElementById("exportSessionBtn");

const timerElement = document.getElementById("timer");
const presentElement = document.getElementById("presentStudents");
const absentElement = document.getElementById("absentStudents");

const attendanceCount =
    document.getElementById("attendanceCount");

const cameraStatus =
    document.getElementById("cameraStatus");

const detectedStudentElement =
    document.getElementById("detectedStudent");

const maskStatusElement =
    document.getElementById("maskStatus");

const livenessStatusElement =
    document.getElementById("livenessStatus");

const confidenceElement =
    document.getElementById("confidence");


function updateCounters() {
    presentElement.textContent = presentStudents;

    if (sessionRunning) {
        absentElement.textContent =
            totalStudents - presentStudents;
    } else {
        absentElement.textContent = 0;
    }

    attendanceCount.textContent =
        `${presentStudents} / ${totalStudents} Present`;
}


function updateTimer() {

    const minutes =
        Math.floor(timeLeft / 60);

    const seconds =
        timeLeft % 60;

    timerElement.textContent =
        `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}


// ------------------------------------------------------------
// Auto-scrolls the attendance list to whichever student is
// currently being detected/verified, and highlights their row
// with a color based on state - so whoever's watching the
// dashboard can see confirmation without manually scrolling
// through all 63 rows.
// ------------------------------------------------------------

const ROW_HIGHLIGHT_COLORS = {
    recognized: "#d4edda",
    already_marked: "#d4edda",
    verifying: "#fff3cd",
    confirming_liveness: "#ffe0f7",
    rejected: "#f8d7da",
    move_into_box: "#e2e3ff",
    look_straight: "#e2e3ff"
};

let lastScrolledEnrollment = null;
let lastHighlightedRow = null;

function scrollToActiveStudent(enrollment, state) {

    if (lastHighlightedRow) {
        lastHighlightedRow.style.backgroundColor = "";
        lastHighlightedRow = null;
    }

    if (!enrollment) {
        lastScrolledEnrollment = null;
        return;
    }

    const rows = document.querySelectorAll(".student-row");

    for (const row of rows) {

        const nameSpan = row.querySelector(".name");

        if (!nameSpan) {
            continue;
        }

        const rowEnrollment =
            nameSpan.textContent.split("-")[0].trim();

        if (rowEnrollment === String(enrollment)) {

            if (enrollment !== lastScrolledEnrollment) {

                row.scrollIntoView({
                    behavior: "smooth",
                    block: "center"
                });

                lastScrolledEnrollment = enrollment;
            }

            row.style.backgroundColor =
                ROW_HIGHLIGHT_COLORS[state] || "#fff3cd";

            lastHighlightedRow = row;

            break;
        }
    }
}


// ------------------------------------------------------------
// Marks a specific student's row as Present, matching against
// the actual dashboard.html markup:
// <div class="student-row">
//     <span class="roll">...</span>
//     <span class="name">2400101460 - MURAD KHAN</span>
//     <span class="attendance-status absent">-</span>
// </div>
// ------------------------------------------------------------

function markStudentPresent(enrollment) {

    const rows = document.querySelectorAll(".student-row");

    rows.forEach((row) => {

        const nameSpan = row.querySelector(".name");

        if (!nameSpan) {
            return;
        }

        const rowEnrollment =
            nameSpan.textContent.split("-")[0].trim();

        if (rowEnrollment === String(enrollment)) {

            const statusSpan =
                row.querySelector(".attendance-status");

            if (statusSpan) {
                statusSpan.textContent = "P";
                statusSpan.classList.remove("absent");
                statusSpan.classList.add("present");
            }
        }
    });
}


// ------------------------------------------------------------
// Clears any "P" markers left over from a previous session, so
// starting a new session doesn't show stale present statuses.
// ------------------------------------------------------------

function resetStudentList() {

    const statusSpans = document.querySelectorAll(
        ".student-row .attendance-status"
    );

    statusSpans.forEach((span) => {
        span.textContent = "-";
        span.classList.remove("present");
        span.classList.add("absent");
    });

    document.querySelectorAll(".student-row").forEach((row) => {
        row.style.backgroundColor = "";
    });

    lastScrolledEnrollment = null;
    lastHighlightedRow = null;
}


// ------------------------------------------------------------
// Shows recognition status in the "Detected Student" /
// "Confidence" boxes below the camera. These are NOT part of
// the video frame, so unlike the text drawn on the camera
// stream, they never get cropped by the camera box's CSS.
// ------------------------------------------------------------

const STATE_LABELS = {
    idle: "---",
    no_face: "No face detected",
    move_into_box: "Move into the box",
    look_straight: "Look more directly at camera",
    confirming_liveness: "Confirming liveness...",
    verifying: "Verifying...",
    recognized: "Recognized",
    rejected: "Unknown / rejected",
    already_marked: "Already marked present"
};

async function pollLiveStatus() {

    try {

        const response = await fetch("/live_status");

        if (!response.ok) {
            return;
        }

        const data = await response.json();

        const stateLabel =
            STATE_LABELS[data.state] || data.state;

        if (data.name) {

            detectedStudentElement.textContent =
                `${stateLabel} - ${data.name} (${data.enrollment})`;

        } else {

            detectedStudentElement.textContent = stateLabel;
        }

        if (data.state === "verifying" || data.state === "recognized") {

            confidenceElement.textContent =
                `${data.candidate_count}/${data.required_confirmations} ` +
                `(gap ${Number(data.gap).toFixed(3)})`;

        } else if (data.distance !== null && data.distance !== undefined) {

            confidenceElement.textContent =
                `distance ${Number(data.distance).toFixed(3)}`;

        } else {

            confidenceElement.textContent = "---";
        }

        if (data.mask_detected === true) {

            maskStatusElement.textContent = "Mask ON";

        } else if (data.mask_detected === false) {

            maskStatusElement.textContent = "No mask";

        } else {

            maskStatusElement.textContent = "---";
        }

        const LIVENESS_LABELS = {
            idle: "---",
            checking: "Checking...",
            live: "Live",
            static: "Hold still / check"
        };

        livenessStatusElement.textContent =
            LIVENESS_LABELS[data.liveness_state] || "---";

        scrollToActiveStudent(data.enrollment, data.state);

    } catch (error) {

        console.error(
            "Failed to fetch live status:",
            error
        );
    }
}


async function pollAttendanceStatus() {

    try {

        const response = await fetch("/attendance_status");

        if (!response.ok) {
            return;
        }

        const data = await response.json();

        totalStudents = data.total;
        presentStudents = data.present;

        updateCounters();

        data.present_enrollments.forEach(markStudentPresent);

    } catch (error) {

        console.error(
            "Failed to fetch attendance status:",
            error
        );
    }
}


async function startAttendance() {

    if (sessionRunning) {
        return;
    }

    let data;

    try {

        const response = await fetch(
            "/start_session",
            { method: "POST" }
        );

        data = await response.json();

    } catch (error) {

        alert("Could not start the attendance session. Please try again.");
        return;
    }

    if (data.status !== "started" && data.status !== "already_active") {
        alert("Could not start the attendance session. Please try again.");
        return;
    }

    if (data.total_students) {
        totalStudents = data.total_students;
    }

    sessionRunning = true;
    presentStudents = 0;

    resetStudentList();

    timeLeft = 10 * 60;
    updateCounters();

    startBtn.disabled = true;
    endBtn.disabled = false;
    if (exportBtn) exportBtn.disabled = true;

    cameraStatus.textContent = "Camera Starting...";

    cameraFeed.src = "/video_feed";
    cameraFeed.style.display = "block";
    document.getElementById("cameraPlaceholder").style.display = "none";

    cameraStatus.textContent = "Camera On";
    updateTimer();

    timerInterval = setInterval(() => {

        timeLeft--;

        updateTimer();

        if (timeLeft <= 0) {
            finishSession();
        }

    }, 1000);

    statusPollInterval = setInterval(
        pollAttendanceStatus,
        2000
    );

    livePollInterval = setInterval(
        pollLiveStatus,
        500
    );
}


async function finishSession() {

    clearInterval(timerInterval);
    clearInterval(statusPollInterval);
    clearInterval(livePollInterval);

    sessionRunning = false;

    cameraFeed.src = "";
    cameraFeed.style.display = "none";
    document.getElementById("cameraPlaceholder").style.display = "flex";

    detectedStudentElement.textContent = "---";
    maskStatusElement.textContent = "---";
    livenessStatusElement.textContent = "---";
    confidenceElement.textContent = "---";

    if (lastHighlightedRow) {
        lastHighlightedRow.style.backgroundColor = "";
        lastHighlightedRow = null;
    }

    lastScrolledEnrollment = null;

    startBtn.disabled = false;
    endBtn.disabled = true;
    if (exportBtn) exportBtn.disabled = false;

    cameraStatus.textContent = "Session Completed";

    try {

        await fetch(
            "/end_session",
            { method: "POST" }
        );

    } catch (error) {

        console.error(
            "Failed to notify backend of session end:",
            error
        );
    }

    // Final refresh so the counters reflect the last confirmed state
    pollAttendanceStatus();

    alert("Attendance session completed!");
}


function endSession() {

    if (!sessionRunning) {
        return;
    }

    const confirmation =
        confirm("Are you sure you want to end the attendance session?");

    if (!confirmation) {
        return;
    }

    finishSession();
}


async function exportLastSession() {

    try {

        const response = await fetch("/export_session");

        if (!response.ok) {

            const data = await response.json().catch(() => ({}));

            if (data.status === "no_session") {
                alert("No completed session yet. End a session first, then export.");
            } else {
                alert("Export failed. Please try again.");
            }

            return;
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);

        let filename = "attendance.xlsx";

        const disposition = response.headers.get("Content-Disposition");

        if (disposition && disposition.includes("filename=")) {
            filename = disposition.split("filename=")[1].replace(/"/g, "").trim();
        }

        const link = document.createElement("a");
        link.href = url;
        link.download = filename;

        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        window.URL.revokeObjectURL(url);

    } catch (error) {

        console.error("Export failed:", error);
        alert("Export failed. Please try again.");
    }
}


startBtn.addEventListener(
    "click",
    startAttendance
);

endBtn.addEventListener(
    "click",
    endSession
);

exportSessionBtn.addEventListener(
    "click",
    exportLastSession
);


updateCounters();
updateTimer();