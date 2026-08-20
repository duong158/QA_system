$ErrorActionPreference = 'Stop'

$outputDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$xlsxPath = (Resolve-Path (Join-Path $outputDir 'ket_qua_100_cau_hoi_ngau_nhien.xlsx')).Path
$mainPreview = Join-Path $outputDir 'preview_ket_qua.png'
$detailPreview = Join-Path $outputDir 'preview_chi_tiet.png'

$excel = $null
$workbook = $null

function Export-RangePreview {
    param(
        [Parameter(Mandatory = $true)] $Worksheet,
        [Parameter(Mandatory = $true)] [string] $RangeAddress,
        [Parameter(Mandatory = $true)] [string] $OutputPath
    )

    $Worksheet.Activate()
    $range = $Worksheet.Range($RangeAddress)
    $range.CopyPicture(1, 2)
    $chartObject = $Worksheet.ChartObjects().Add(0, 0, $range.Width, $range.Height)
    try {
        $chartObject.Chart.Paste()
        if (-not $chartObject.Chart.Export($OutputPath, 'PNG', $false)) {
            throw "Excel could not export preview: $OutputPath"
        }
    }
    finally {
        $chartObject.Delete()
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($chartObject)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($range)
    }
}

try {
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $excel.ScreenUpdating = $false
    $excel.EnableEvents = $false
    $missing = [Type]::Missing
    $workbook = $excel.Workbooks.Open(
        $xlsxPath, 0, $false, $missing, $missing, $missing, $true, $missing,
        $missing, $false, $false, $missing, $false, $true, 1
    )
    $excel.CalculateFullRebuild()
    $workbook.Save()

    $mainSheet = $workbook.Worksheets.Item('Kết quả QA')
    $detailSheet = $workbook.Worksheets.Item('Chi tiết kỹ thuật')
    try {
        Export-RangePreview -Worksheet $mainSheet -RangeAddress 'A1:H18' -OutputPath $mainPreview
        Export-RangePreview -Worksheet $detailSheet -RangeAddress 'A1:O12' -OutputPath $detailPreview
    }
    finally {
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($mainSheet)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($detailSheet)
    }
}
finally {
    if ($workbook -ne $null) {
        $workbook.Close($true)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($workbook)
    }
    if ($excel -ne $null) {
        $excel.Quit()
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($excel)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

Get-Item -LiteralPath $mainPreview, $detailPreview | Select-Object FullName, Length, LastWriteTime
