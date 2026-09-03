// Import_Bruker_2dseq.ijm
//
// Open a Bruker ParaVision reconstruction (pdata/N/2dseq) in ImageJ with the
// parameters read from visu_pars: data type, matrix, byte order, number of
// frames, voxel size, frame-group layout (slices / echoes / diffusion / time)
// as a hyperstack, and optional intensity scaling (VisuCoreDataSlope/Offs).
//
// Install: copy into <ImageJ>/plugins and restart ImageJ, then use
//   Plugins > Import Bruker 2dseq        (or  brukerview install-imagej)
// You can point it at a study folder (pick a scan from a list), a scan folder,
// or a pdata/N folder.
//
// Command line:
//   ImageJ.exe -macro Import_Bruker_2dseq.ijm "E:\data\study\5\pdata\1"
//   Options appended to the argument with "|":
//     raw     keep the stored 16-bit integers (no intensity scaling)
//     nobc    do not open the Brightness/Contrast panel
//     nozoom  leave the window at 100% (used by headless tests)
//   e.g. "E:\data\study\5\pdata\1|raw|nobc"
//
// Display: the range is set from the whole stack's histogram (0.35% saturated)
// so every slice/echo/direction is visible; Ctrl+Shift+C toggles the
// Brightness/Contrast panel, whose Auto button re-stretches the current plane.
//
// Macro-language notes (all of these fail silently or with an error dialog):
//  - functions only see globals declared with `var`
//  - a user function's result must be stored in a variable before it is
//    returned or passed to another function
//  - `return` and `=` only parse arithmetic: put comparisons and && / || in an
//    if statement and return true/false explicitly

requires("1.47a");

var batch = false;        // true when a path argument was given: no dialogs
var applyScale = true;
var openBC = true;        // open Brightness/Contrast panel after import
var autoZoom = true;      // enlarge small image windows
var stackRange = true;    // display range from the whole stack, not plane 1

arg = getArgument();
if (arg != "") {
    parts = split(arg, "|");
    dir = parts[0];
    for (i = 1; i < parts.length; i++) {
        if (parts[i] == "raw") applyScale = false;
        if (parts[i] == "nobc") openBC = false;
        if (parts[i] == "nozoom") autoZoom = false;
    }
    batch = true;
} else {
    dir = getDirectory("Choose a Bruker study, scan, or pdata folder");
    if (dir == "") exit();
}
dir = withSep(dir);
pdata = resolvePdata(dir);
if (pdata == "") exit("No Bruker reconstruction (visu_pars + 2dseq) found under\n" + dir);

if (!batch) {
    Dialog.create("Import Bruker 2dseq");
    Dialog.addMessage(pdata);
    Dialog.addCheckbox("Apply intensity scaling (converts to 32-bit)", applyScale);
    Dialog.addCheckbox("Display range from whole stack (else from first plane)", stackRange);
    Dialog.addCheckbox("Open Brightness/Contrast panel", openBC);
    Dialog.show();
    applyScale = Dialog.getCheckbox();
    stackRange = Dialog.getCheckbox();
    openBC = Dialog.getCheckbox();
}
openBruker(pdata, applyScale);


// ---------------------------------------------------------------------------
function openBruker(pdata, applyScale) {
    visu = File.openAsString(pdata + "visu_pars");
    visu = replace(visu, "\r", "");

    v = visuValue(visu, "VisuCoreDim");
    dim = parseInt(v);
    if (isNaN(dim) || dim < 2) exit("Not an image dataset (VisuCoreDim=" + v + "). Spectroscopy is not supported.");
    v = visuValue(visu, "VisuCoreSize");   size = numArray(v);
    v = visuValue(visu, "VisuCoreExtent"); extent = numArray(v);
    v = visuValue(visu, "VisuCoreFrameCount"); frames = parseInt(v);
    if (isNaN(frames)) frames = 1;
    v = visuValue(visu, "VisuCoreWordType");  wordType = bvTrim(v);
    v = visuValue(visu, "VisuCoreByteOrder"); byteOrder = bvTrim(v);
    protocol = visuStr(visu, "VisuAcquisitionProtocol");

    nx = size[0]; ny = size[1];
    nz = 1;
    if (dim >= 3) nz = size[2];
    nImagesExpected = frames * nz;

    ijType = "";
    if (wordType == "_16BIT_SGN_INT") ijType = "16-bit Signed";
    else if (wordType == "_16BIT_UNSGN_INT") ijType = "16-bit Unsigned";
    else if (wordType == "_8BIT_UNSGN_INT") ijType = "8-bit";
    else if (wordType == "_32BIT_SGN_INT") ijType = "32-bit Signed";
    else if (wordType == "_32BIT_UNSGN_INT") ijType = "32-bit Unsigned";
    else if (wordType == "_32BIT_FLOAT") ijType = "32-bit Real";
    else if (wordType == "_64BIT_FLOAT") ijType = "64-bit Real";
    else exit("Unsupported VisuCoreWordType: " + wordType);

    opts = "open=[" + pdata + "2dseq] image=[" + ijType + "] width=" + nx + " height=" + ny
         + " offset=0 number=" + nImagesExpected + " gap=0";
    if (byteOrder == "littleEndian") opts = opts + " little-endian";
    setBatchMode(true);
    run("Raw...", opts);

    // ---- intensity scaling -------------------------------------------------
    if (applyScale) {
        v = visuValue(visu, "VisuCoreDataSlope"); slope = numArray(v);
        v = visuValue(visu, "VisuCoreDataOffs");  offs = numArray(v);
        if (slope.length == 0) slope = newArray(1);
        if (offs.length == 0) offs = newArray(0);
        run("32-bit");
        sameSlope = bvUniform(slope);
        sameOffs = bvUniform(offs);
        if (sameSlope && sameOffs) {
            if (slope[0] != 1) run("Multiply...", "value=" + slope[0] + " stack");
            if (offs[0] != 0) run("Add...", "value=" + offs[0] + " stack");
        } else {
            for (f = 0; f < frames; f++) {
                s = slope[minOf(f, slope.length - 1)];
                o = offs[minOf(f, offs.length - 1)];
                for (k = 0; k < nz; k++) {
                    setSlice(f * nz + k + 1);
                    if (s != 1) run("Multiply...", "value=" + s + " slice");
                    if (o != 0) run("Add...", "value=" + o + " slice");
                }
            }
            setSlice(1);
        }
    }

    // ---- frame groups -> hyperstack ---------------------------------------------
    // VisuFGOrderDesc lists groups fastest-varying first; ImageJ orders c, z, t.
    gSizes = newArray(0);
    gNames = newArray(0);
    v = visuValue(visu, "VisuFGOrderDesc");
    fg = stripDims(v);
    p = indexOf(fg, "(");
    while (p >= 0) {
        q = indexOf(fg, ")", p);
        if (q < 0) q = lengthOf(fg);
        fields = split(substring(fg, p + 1, q), ",");
        if (fields.length >= 2) {
            t = bvTrim(fields[0]);
            n = parseInt(t);
            nm = strValue(fields[1]);
            gSizes = Array.concat(gSizes, n);
            gNames = Array.concat(gNames, nm);
        }
        p = indexOf(fg, "(", q);
    }
    prod = 1;
    for (i = 0; i < gSizes.length; i++) prod = prod * gSizes[i];
    if (gSizes.length == 0 || prod != frames) {
        gSizes = newArray(1); gSizes[0] = frames;
        gNames = newArray(1);
        if (dim == 2) gNames[0] = "FG_SLICE"; else gNames[0] = "FG_FRAME";
    }

    // Assign each frame group a hyperstack axis: slices -> z, the first other
    // group -> t, a second other group -> c. "Stack to Hyperstack" takes the
    // input order as a string listing axes fastest-first, which is exactly how
    // VisuFGOrderDesc lists the groups, so no separate re-ordering is needed.
    layout = "";
    if (dim >= 3) {
        if (frames > 1) {
            run("Stack to Hyperstack...", "order=xyczt(default) channels=1 slices=" + nz + " frames=" + frames + " display=Grayscale");
            lab = groupLabel(gNames[0]);
            layout = "" + nz + " slices x " + frames + " " + lab;
        } else {
            layout = "" + nz + " slices";
        }
    } else if (gSizes.length == 1) {
        lab = groupLabel(gNames[0]);
        layout = "" + gSizes[0] + " " + lab;
    } else if (gSizes.length <= 3) {
        sliceSlot = -1;
        for (i = 0; i < gSizes.length; i++) if (gNames[i] == "FG_SLICE") sliceSlot = i;
        if (sliceSlot < 0) sliceSlot = 0;
        nC = 1; nZ = 1; nT = 1;
        order = "xy";
        used = "";
        layout = "";
        for (i = 0; i < gSizes.length; i++) {
            lab = groupLabel(gNames[i]);
            if (i == sliceSlot) { axis = "z"; nZ = gSizes[i]; }
            else if (indexOf(used, "t") < 0) { axis = "t"; nT = gSizes[i]; }
            else { axis = "c"; nC = gSizes[i]; }
            used = used + axis;
            order = order + axis;
            if (layout != "") layout = layout + " x ";
            layout = layout + gSizes[i] + " " + lab + " (" + axis + ")";
        }
        // pad with the unused axes so the order string is a full permutation
        if (indexOf(used, "c") < 0) order = order + "c";
        if (indexOf(used, "z") < 0) order = order + "z";
        if (indexOf(used, "t") < 0) order = order + "t";
        if (order == "xyczt") order = "xyczt(default)";
        run("Stack to Hyperstack...", "order=" + order + " channels=" + nC + " slices=" + nZ + " frames=" + nT + " display=Grayscale");
    } else {
        layout = "" + frames + " frames (" + gSizes.length + " frame groups, left as a plain stack)";
    }

    // ---- calibration and metadata ---------------------------------------------------
    dx = extent[0] / nx;
    dy = extent[1] / ny;
    if (dim >= 3) {
        dz = extent[2] / nz;
    } else {
        v = visuValue(visu, "VisuCoreSlicePacksSliceDist"); dz = firstNum(v);
        if (isNaN(dz)) { v = visuValue(visu, "VisuCoreFrameThickness"); dz = firstNum(v); }
        if (isNaN(dz)) dz = 1;
    }
    setVoxelSize(dx, dy, dz, "mm");

    pdataNoSep = noSep(pdata);
    procNo = File.getName(pdataNoSep);
    pdataParent = File.getParent(pdataNoSep);
    scanDir = File.getParent(pdataParent);
    scanName = File.getName(scanDir);
    method = "";
    methodPath = scanDir + File.separator + "method";
    if (File.exists(methodPath)) {
        m = File.openAsString(methodPath);
        m = replace(m, "\r", "");
        method = visuStr(m, "Method");
    }
    v = visuValue(visu, "VisuAcqEchoTime");       te = stripDims(v); te = bvTrim(te);
    v = visuValue(visu, "VisuAcqRepetitionTime"); tr = stripDims(v); tr = bvTrim(tr);
    subject = visuStr(visu, "VisuSubjectId");
    study = visuStr(visu, "VisuStudyId");
    fgText = bvTrim(fg);
    info = "Bruker 2dseq import\n"
         + "Path: " + pdata + "\n"
         + "Protocol: " + protocol + "\n"
         + "Method: " + method + "\n"
         + "Subject: " + subject + "\n"
         + "Study: " + study + "\n"
         + "Matrix: " + nx + " x " + ny + " x " + nz + "   frames: " + frames + "\n"
         + "Layout: " + layout + "\n"
         + "Voxel (mm): " + d2s(dx, 4) + " x " + d2s(dy, 4) + " x " + d2s(dz, 4) + "\n"
         + "TE (ms): " + te + "\nTR (ms): " + tr + "\n"
         + "Word type: " + wordType + "   scaled: " + applyScale + "\n"
         + "Frame groups: " + fgText + "\n";
    setMetadata("Info", info);
    rename(scanName + "_" + protocol + " [p" + procNo + "]");
    // display range: robust stretch over the whole stack so dark frames
    // (diffusion-weighted, late echoes) are not crushed by a bright first plane
    resetMinAndMax();
    if (stackRange && nSlices > 1) run("Enhance Contrast...", "saturated=0.35 use");
    else run("Enhance Contrast", "saturated=0.35");
    setBatchMode("exit and display");
    if (autoZoom) {
        // grow small windows (a 128 px matrix opens as a tiny window otherwise)
        n = 0;
        while (getZoom() * getWidth() < 600 && n < 6) { run("In [+]"); n++; }
    }
    if (openBC) run("Brightness/Contrast...");
}

// ---------------------------------------------------------------------------
// Locate a pdata folder from whatever the user pointed at.
function resolvePdata(d) {
    ok = isPdata(d);
    if (ok) return d;
    ok = isScan(d);
    if (ok) { r = choosePdata(d); return r; }
    scans = listScans(d);
    if (scans.length == 0) {
        // zip extraction often wraps the study in one extra folder
        subs = listDirs(d);
        for (i = 0; i < subs.length && scans.length == 0; i++) {
            inner = listScans(d + subs[i]);
            if (inner.length > 0) { d = d + subs[i]; scans = inner; }
        }
    }
    if (scans.length == 0) return "";
    if (batch) { r = choosePdata(d + scans[0]); return r; }
    labels = newArray(scans.length);
    for (i = 0; i < scans.length; i++) {
        lab = noSep(scans[i]);
        procs = listPdata(d + scans[i]);
        if (procs.length > 0) {
            v = File.openAsString(d + scans[i] + procs[0] + "visu_pars");
            v = replace(v, "\r", "");
            proto = visuStr(v, "VisuAcquisitionProtocol");
            sz = visuValue(v, "VisuCoreSize"); sz = stripDims(sz); sz = bvTrim(sz); sz = replace(sz, " ", "x");
            fr = visuValue(v, "VisuCoreFrameCount");
            lab = lab + "   " + proto + "   " + sz + "   " + fr + " fr";
        }
        labels[i] = lab;
    }
    Dialog.create("Import Bruker 2dseq");
    Dialog.addMessage(d);
    Dialog.addChoice("Scan", labels, labels[0]);
    Dialog.show();
    pick = Dialog.getChoice();
    for (i = 0; i < scans.length; i++) {
        if (labels[i] == pick) { r = choosePdata(d + scans[i]); return r; }
    }
    return "";
}

function choosePdata(scan) {
    procs = listPdata(scan);
    if (procs.length == 0) return "";
    if (procs.length == 1 || batch) { r = scan + procs[0]; return r; }
    Dialog.create("Import Bruker 2dseq");
    Dialog.addChoice("Reconstruction (pdata)", procs, procs[0]);
    Dialog.show();
    r = Dialog.getChoice();
    r = scan + r;
    return r;
}

// Note: `return a && b` or `return x > y` do not work in the macro language
// (return only parses arithmetic), hence the explicit if statements.
function isPdata(d) {
    a = File.exists(d + "visu_pars");
    b = File.exists(d + "2dseq");
    if (a && b) return true;
    return false;
}
function isScan(d) {
    a = File.isDirectory(d + "pdata");
    b = File.exists(d + "acqp");
    c = File.exists(d + "method");
    if (a && (b || c)) return true;
    return false;
}

function listDirs(d) {
    list = getFileList(d);
    out = newArray(0);
    for (i = 0; i < list.length; i++) if (endsWith(list[i], "/")) out = Array.concat(out, list[i]);
    sortNatural(out);
    return out;
}
function listScans(d) {
    subs = listDirs(d);
    out = newArray(0);
    for (i = 0; i < subs.length; i++) {
        ok = isScan(d + subs[i]);
        if (ok) out = Array.concat(out, subs[i]);
    }
    return out;
}
function listPdata(scan) {
    subs = listDirs(scan + "pdata/");
    out = newArray(0);
    for (i = 0; i < subs.length; i++) {
        ok = isPdata(scan + "pdata/" + subs[i]);
        if (ok) out = Array.concat(out, "pdata/" + subs[i]);
    }
    return out;
}

// numeric folder names (1, 2, ..., 10) in numeric order, then the rest alphabetically
function sortNatural(a) {
    n = a.length;
    for (i = 1; i < n; i++) {
        v = a[i];
        j = i - 1;
        moving = true;
        while (j >= 0 && moving) {
            g = natGreater(a[j], v);
            if (g) { a[j + 1] = a[j]; j--; }
            else moving = false;
        }
        a[j + 1] = v;
    }
}
// numeric names sort by value and come first; other names keep the
// alphabetical order getFileList already gives (insertion sort is stable)
function natGreater(x, y) {
    sx = noSep(x); sy = noSep(y);
    nx = parseInt(sx); ny = parseInt(sy);
    if (!isNaN(nx) && !isNaN(ny)) {
        if (nx > ny) return true;
        return false;
    }
    if (!isNaN(nx)) return false;
    if (!isNaN(ny)) return true;
    return false;
}

// ---- JCAMP-DX helpers ----------------------------------------------------------
// Raw text of "##$key=..." including continuation lines, without $$ comments.
function visuValue(text, key) {
    tag = "##$" + key + "=";
    i = indexOf(text, "\n" + tag);
    if (i >= 0) i = i + 1;
    else if (startsWith(text, tag)) i = 0;
    else return "";
    start = i + lengthOf(tag);
    end = indexOf(text, "\n##", start);
    if (end < 0) end = lengthOf(text);
    lines = split(substring(text, start, end), "\n");
    out = "";
    for (k = 0; k < lines.length; k++) if (!startsWith(lines[k], "$$")) out = out + " " + lines[k];
    out = bvTrim(out);
    return out;
}
// "<string>" value of a key
function visuStr(text, key) {
    v = visuValue(text, key);
    s = strValue(v);
    return s;
}
function stripDims(v) { return replace(v, "^\\s*\\(\\s*[0-9, ]+\\)", ""); }
function strValue(v) {
    a = indexOf(v, "<"); b = indexOf(v, ">");
    if (a >= 0 && b > a) return substring(v, a + 1, b);
    s = bvTrim(v);
    return s;
}
function numArray(v) {
    v = stripDims(v);
    v = bvTrim(v);
    out = newArray(0);
    if (v == "") return out;
    toks = split(v, " ");
    for (i = 0; i < toks.length; i++) {
        t = toks[i];
        if (startsWith(t, "@")) {          // @29*(0.877) run-length shorthand
            n = parseInt(substring(t, 1, indexOf(t, "*")));
            val = parseFloat(substring(t, indexOf(t, "(") + 1, indexOf(t, ")")));
            rep = newArray(n);
            Array.fill(rep, val);
            out = Array.concat(out, rep);
        } else {
            val = parseFloat(t);
            out = Array.concat(out, val);
        }
    }
    return out;
}
function firstNum(v) {
    a = numArray(v);
    if (a.length == 0) return NaN;
    return a[0];
}
function bvUniform(a) {
    for (i = 1; i < a.length; i++) if (a[i] != a[0]) return false;
    return true;
}
function bvTrim(s) { return replace(s, "^\\s+|\\s+$", ""); }
function withSep(d) {
    if (endsWith(d, "/") || endsWith(d, "\\")) return d;
    d = d + File.separator;
    return d;
}
function noSep(d) {
    if (endsWith(d, "/") || endsWith(d, "\\")) return substring(d, 0, lengthOf(d) - 1);
    return d;
}
function groupLabel(name) {
    if (name == "FG_SLICE") return "slices";
    s = replace(name, "^FG_", "");
    s = toLowerCase(s);
    return s;
}
