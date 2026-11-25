' Jalankan aplikasi tanpa jendela terminal.
' Virtual environment dicari di folder aplikasi; tidak bergantung pada folder
' atau struktur direktori komputer tertentu.
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")

here = fso.GetParentFolderName(WScript.ScriptFullName)
pyw = here & "\.venv\Scripts\pythonw.exe"
If Not fso.FileExists(pyw) Then pyw = here & "\venv\Scripts\pythonw.exe"

If fso.FileExists(pyw) Then
    sh.CurrentDirectory = here
    sh.Run """" & pyw & """ """" & here & "\app.py"""", 0, False
Else
    MsgBox "Virtual environment tidak ditemukan. Buat .venv di folder aplikasi dan pasang requirements.txt. Lihat README.md untuk langkah instalasi.", vbExclamation, "UI-Teaching"
    WScript.Quit 1
End If
