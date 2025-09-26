package com.example.nsfw_schutz

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.nsfw_schutz.R
import com.example.nsfw_schutz.ui.theme.NsfwschutzTheme
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private const val BASE_URL = "http://45.147.7.198:7535"

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            NsfwschutzTheme {
                MainScreen()
            }
        }
    }
}

data class ModelInfo(
    val slug: String,
    val displayName: String,
    val version: Int?,
    val tier: String
)

data class AgentOption(
    val count: Int,
    val headline: String,
    val description: String
)

data class BackendStatus(
    val label: String,
    val message: String,
    val timestamp: Long,
    val type: StatusType
)

enum class StatusType {
    SUCCESS,
    WARNING,
    ERROR
}

private val fallbackModels = listOf(
    ModelInfo("nexia_fast", "nexia fast version 1", 1, "fast"),
    ModelInfo("nexia_ai_pro", "nexia ai pro version 1", 1, "pro"),
    ModelInfo("nexia_ai_lite", "nexia ai lite version 1", 1, "lite")
)

private val agentOptions = listOf(
    AgentOption(1, "1 Agent", "Perfekt bei schwacher Verbindung – Bilder werden nacheinander geprüft."),
    AgentOption(2, "2 Agenten", "Guter Ausgleich aus Geschwindigkeit und Serverlast."),
    AgentOption(3, "3 Agenten", "Schnelle Prüfungen, benötigt bereits mehr Leistung auf dem Gerät."),
    AgentOption(4, "4 Agenten (Maximum)", "Maximale Geschwindigkeit, höhere Rechenlast & Serverkosten.")
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen() {
    val coroutineScope = rememberCoroutineScope()
    var connectionState by rememberSaveable { mutableStateOf<BackendStatus?>(null) }
    var models by rememberSaveable { mutableStateOf(fallbackModels) }
    var selectedModelSlug by rememberSaveable { mutableStateOf(fallbackModels.first().slug) }
    var isRefreshing by remember { mutableStateOf(false) }

    var agentCount by rememberSaveable { mutableStateOf(2f) }
    var autoScanEnabled by rememberSaveable { mutableStateOf(true) }
    var queueFallbackEnabled by rememberSaveable { mutableStateOf(true) }
    var autoScanAlwaysOn by rememberSaveable { mutableStateOf(true) }
    var scheduleEnabled by rememberSaveable { mutableStateOf(false) }
    var scheduleStart by rememberSaveable { mutableStateOf("06:00") }
    var scheduleEnd by rememberSaveable { mutableStateOf("22:00") }
    var deleteImmediately by rememberSaveable { mutableStateOf(false) }
    var quarantineFolder by rememberSaveable { mutableStateOf("NSFW Erkennung") }
    var parentalLockEnabled by rememberSaveable { mutableStateOf(false) }
    var parentalPassword by rememberSaveable { mutableStateOf("") }
    var parentalPasswordRepeat by rememberSaveable { mutableStateOf("") }
    var queueLimit by rememberSaveable { mutableStateOf(50f) }

    LaunchedEffect(Unit) {
        isRefreshing = true
        connectionState = loadBackendStatus()
        val modelResult = loadModels()
        models = modelResult.ifEmpty { fallbackModels }
        if (models.none { it.slug == selectedModelSlug }) {
            selectedModelSlug = models.firstOrNull()?.slug ?: ""
        }
        isRefreshing = false
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(text = stringResource(id = R.string.dashboard_title)) }
            )
        }
    ) { innerPadding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            item {
                StatusCard(
                    connectionState = connectionState,
                    isRefreshing = isRefreshing,
                    onRefresh = {
                        coroutineScope.launch {
                            isRefreshing = true
                            connectionState = loadBackendStatus()
                            isRefreshing = false
                        }
                    }
                )
            }

            item {
                ModelCard(
                    models = models,
                    selectedModelSlug = selectedModelSlug,
                    onSelect = { selectedModelSlug = it },
                    onRefresh = {
                        coroutineScope.launch {
                            isRefreshing = true
                            val newModels = loadModels()
                            models = newModels.ifEmpty { fallbackModels }
                            if (models.none { it.slug == selectedModelSlug }) {
                                selectedModelSlug = models.firstOrNull()?.slug ?: selectedModelSlug
                            }
                            isRefreshing = false
                        }
                    },
                    isRefreshing = isRefreshing
                )
            }

            item {
                AgentCard(
                    agentCount = agentCount,
                    onAgentChange = { agentCount = it },
                    queueLimit = queueLimit,
                    onQueueLimitChange = { queueLimit = it },
                    queueFallbackEnabled = queueFallbackEnabled,
                    onQueueFallbackChange = { queueFallbackEnabled = it }
                )
            }

            item {
                AutomationCard(
                    autoScanEnabled = autoScanEnabled,
                    onAutoScanChange = { autoScanEnabled = it },
                    autoScanAlwaysOn = autoScanAlwaysOn,
                    onAlwaysOnChange = { autoScanAlwaysOn = it },
                    scheduleEnabled = scheduleEnabled,
                    onScheduleChange = { scheduleEnabled = it },
                    scheduleStart = scheduleStart,
                    onScheduleStartChange = { scheduleStart = it },
                    scheduleEnd = scheduleEnd,
                    onScheduleEndChange = { scheduleEnd = it }
                )
            }

            item {
                SafetyCard(
                    deleteImmediately = deleteImmediately,
                    onDeleteImmediatelyChange = { deleteImmediately = it },
                    quarantineFolder = quarantineFolder,
                    onQuarantineChange = { quarantineFolder = it }
                )
            }

            item {
                ParentalLockCard(
                    enabled = parentalLockEnabled,
                    onEnabledChange = { parentalLockEnabled = it },
                    password = parentalPassword,
                    onPasswordChange = { parentalPassword = it },
                    confirmPassword = parentalPasswordRepeat,
                    onConfirmPasswordChange = { parentalPasswordRepeat = it }
                )
            }

            item {
                SupportCard()
            }
        }
    }
}

@Composable
private fun StatusCard(
    connectionState: BackendStatus?,
    isRefreshing: Boolean,
    onRefresh: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = stringResource(id = R.string.status_section_title),
                style = MaterialTheme.typography.titleMedium
            )
            Spacer(modifier = Modifier.height(8.dp))
            val state = connectionState
            if (state == null) {
                Text(text = stringResource(id = R.string.status_initial_loading))
            } else {
                val statusColor = when (state.type) {
                    StatusType.SUCCESS -> MaterialTheme.colorScheme.primary
                    StatusType.WARNING -> MaterialTheme.colorScheme.secondary
                    StatusType.ERROR -> MaterialTheme.colorScheme.error
                }
                Text(
                    text = state.label,
                    style = MaterialTheme.typography.titleLarge,
                    color = statusColor
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(text = state.message)
                Spacer(modifier = Modifier.height(4.dp))
                val dateFormat = remember { SimpleDateFormat("dd.MM.yyyy HH:mm:ss", Locale.getDefault()) }
                Text(
                    text = stringResource(
                        id = R.string.status_last_checked,
                        dateFormat.format(Date(state.timestamp))
                    ),
                    style = MaterialTheme.typography.labelSmall
                )
            }
            Spacer(modifier = Modifier.height(12.dp))
            Button(onClick = onRefresh, enabled = !isRefreshing) {
                Text(text = if (isRefreshing) stringResource(id = R.string.status_refreshing) else stringResource(id = R.string.status_refresh))
            }
        }
    }
}

@Composable
private fun ModelCard(
    models: List<ModelInfo>,
    selectedModelSlug: String,
    onSelect: (String) -> Unit,
    onRefresh: () -> Unit,
    isRefreshing: Boolean
) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = stringResource(id = R.string.model_section_title),
                style = MaterialTheme.typography.titleMedium
            )
            Spacer(modifier = Modifier.height(8.dp))
            if (models.isEmpty()) {
                Text(text = stringResource(id = R.string.model_empty))
            } else {
                models.forEach { model ->
                    ModelRow(
                        model = model,
                        selected = model.slug == selectedModelSlug,
                        onSelect = { onSelect(model.slug) }
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedButton(onClick = onRefresh, enabled = !isRefreshing) {
                Text(text = if (isRefreshing) stringResource(id = R.string.model_refreshing) else stringResource(id = R.string.model_refresh))
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ModelRow(
    model: ModelInfo,
    selected: Boolean,
    onSelect: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (selected) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant
        ),
        onClick = onSelect
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Text(
                text = model.displayName,
                style = MaterialTheme.typography.bodyLarge,
                color = if (selected) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(4.dp))
            val tierDescription = when (model.tier.lowercase(Locale.getDefault())) {
                "fast" -> stringResource(id = R.string.model_tier_fast)
                "pro" -> stringResource(id = R.string.model_tier_pro)
                "lite" -> stringResource(id = R.string.model_tier_lite)
                else -> stringResource(id = R.string.model_tier_default)
            }
            Text(
                text = tierDescription,
                style = MaterialTheme.typography.bodySmall
            )
            model.version?.let {
                Text(
                    text = stringResource(id = R.string.model_version, it),
                    style = MaterialTheme.typography.labelSmall
                )
            }
            AnimatedVisibility(visible = selected) {
                Text(
                    text = stringResource(id = R.string.model_selected_hint),
                    style = MaterialTheme.typography.labelSmall,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }
        }
    }
}

@Composable
private fun AgentCard(
    agentCount: Float,
    onAgentChange: (Float) -> Unit,
    queueLimit: Float,
    onQueueLimitChange: (Float) -> Unit,
    queueFallbackEnabled: Boolean,
    onQueueFallbackChange: (Boolean) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = stringResource(id = R.string.agent_section_title),
                style = MaterialTheme.typography.titleMedium
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(text = stringResource(id = R.string.agent_section_description))
            Spacer(modifier = Modifier.height(12.dp))
            Slider(
                value = agentCount,
                onValueChange = { value -> onAgentChange(value.coerceIn(1f, 4f)) },
                valueRange = 1f..4f,
                steps = 2,
                colors = SliderDefaults.colors(
                    thumbColor = MaterialTheme.colorScheme.primary,
                    activeTrackColor = MaterialTheme.colorScheme.primary
                )
            )
            val rounded = agentCount.toInt()
            val agentInfo = agentOptions.firstOrNull { it.count == rounded }
            Text(
                text = agentInfo?.headline ?: stringResource(id = R.string.agent_unknown_count, rounded),
                style = MaterialTheme.typography.titleSmall
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(text = agentInfo?.description ?: "")
            Spacer(modifier = Modifier.height(12.dp))
            Text(text = stringResource(id = R.string.agent_queue_limit, queueLimit.toInt()))
            Slider(
                value = queueLimit,
                onValueChange = { onQueueLimitChange(it.coerceIn(10f, 200f)) },
                valueRange = 10f..200f,
                steps = 18
            )
            Spacer(modifier = Modifier.height(8.dp))
            Row(
                verticalAlignment = Alignment.CenterVertically
            ) {
                Switch(
                    checked = queueFallbackEnabled,
                    onCheckedChange = onQueueFallbackChange,
                    colors = SwitchDefaults.colors(checkedThumbColor = MaterialTheme.colorScheme.primary)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Column {
                    Text(text = stringResource(id = R.string.agent_queue_fallback_title))
                    Text(
                        text = stringResource(id = R.string.agent_queue_fallback_description),
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
        }
    }
}

@Composable
private fun AutomationCard(
    autoScanEnabled: Boolean,
    onAutoScanChange: (Boolean) -> Unit,
    autoScanAlwaysOn: Boolean,
    onAlwaysOnChange: (Boolean) -> Unit,
    scheduleEnabled: Boolean,
    onScheduleChange: (Boolean) -> Unit,
    scheduleStart: String,
    onScheduleStartChange: (String) -> Unit,
    scheduleEnd: String,
    onScheduleEndChange: (String) -> Unit
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = stringResource(id = R.string.automation_section_title),
                style = MaterialTheme.typography.titleMedium
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(text = stringResource(id = R.string.automation_description))
            Spacer(modifier = Modifier.height(12.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Switch(
                    checked = autoScanEnabled,
                    onCheckedChange = onAutoScanChange,
                    colors = SwitchDefaults.colors(checkedThumbColor = MaterialTheme.colorScheme.primary)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(text = stringResource(id = R.string.automation_enable_label))
            }
            AnimatedVisibility(visible = autoScanEnabled) {
                Column {
                    Spacer(modifier = Modifier.height(12.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Switch(
                            checked = autoScanAlwaysOn,
                            onCheckedChange = onAlwaysOnChange
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(text = stringResource(id = R.string.automation_24_7_label))
                    }
                    AnimatedVisibility(visible = !autoScanAlwaysOn) {
                        Column {
                            Spacer(modifier = Modifier.height(12.dp))
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Switch(
                                    checked = scheduleEnabled,
                                    onCheckedChange = onScheduleChange
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(text = stringResource(id = R.string.automation_schedule_label))
                            }
                            AnimatedVisibility(visible = scheduleEnabled) {
                                Column {
                                    Spacer(modifier = Modifier.height(8.dp))
                                    ScheduleField(
                                        label = stringResource(id = R.string.automation_schedule_start),
                                        value = scheduleStart,
                                        onValueChange = onScheduleStartChange
                                    )
                                    Spacer(modifier = Modifier.height(8.dp))
                                    ScheduleField(
                                        label = stringResource(id = R.string.automation_schedule_end),
                                        value = scheduleEnd,
                                        onValueChange = onScheduleEndChange
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ScheduleField(
    label: String,
    value: String,
    onValueChange: (String) -> Unit
) {
    OutlinedTextField(
        value = value,
        onValueChange = { newValue ->
            if (newValue.length <= 5) {
                onValueChange(newValue)
            }
        },
        label = { Text(text = label) },
        placeholder = { Text(text = "HH:MM") },
        singleLine = true,
        supportingText = {
            Text(text = stringResource(id = R.string.automation_schedule_hint))
        }
    )
}

@Composable
private fun SafetyCard(
    deleteImmediately: Boolean,
    onDeleteImmediatelyChange: (Boolean) -> Unit,
    quarantineFolder: String,
    onQuarantineChange: (String) -> Unit
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = stringResource(id = R.string.safety_section_title),
                style = MaterialTheme.typography.titleMedium
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(text = stringResource(id = R.string.safety_description))
            Spacer(modifier = Modifier.height(12.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Switch(
                    checked = deleteImmediately,
                    onCheckedChange = onDeleteImmediatelyChange
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(text = stringResource(id = R.string.safety_delete_label))
            }
            AnimatedVisibility(visible = !deleteImmediately) {
                Column {
                    Spacer(modifier = Modifier.height(12.dp))
                    OutlinedTextField(
                        value = quarantineFolder,
                        onValueChange = onQuarantineChange,
                        label = { Text(text = stringResource(id = R.string.safety_quarantine_label)) },
                        singleLine = true
                    )
                }
            }
        }
    }
}

@Composable
private fun ParentalLockCard(
    enabled: Boolean,
    onEnabledChange: (Boolean) -> Unit,
    password: String,
    onPasswordChange: (String) -> Unit,
    confirmPassword: String,
    onConfirmPasswordChange: (String) -> Unit
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = stringResource(id = R.string.parental_section_title),
                style = MaterialTheme.typography.titleMedium
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(text = stringResource(id = R.string.parental_description))
            Spacer(modifier = Modifier.height(12.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Switch(
                    checked = enabled,
                    onCheckedChange = onEnabledChange
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(text = stringResource(id = R.string.parental_enable_label))
            }
            AnimatedVisibility(visible = enabled) {
                Column {
                    Spacer(modifier = Modifier.height(12.dp))
                    SecureTextField(
                        value = password,
                        onValueChange = onPasswordChange,
                        label = stringResource(id = R.string.parental_password_label)
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    SecureTextField(
                        value = confirmPassword,
                        onValueChange = onConfirmPasswordChange,
                        label = stringResource(id = R.string.parental_password_repeat)
                    )
                    if (password.isNotEmpty() && confirmPassword.isNotEmpty() && password != confirmPassword) {
                        Text(
                            text = stringResource(id = R.string.parental_password_mismatch),
                            color = MaterialTheme.colorScheme.error,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.padding(top = 4.dp)
                        )
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = stringResource(id = R.string.parental_note),
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
        }
    }
}

@Composable
private fun SecureTextField(
    value: String,
    onValueChange: (String) -> Unit,
    label: String
) {
    var isVisible by rememberSaveable { mutableStateOf(false) }
    Column {
        OutlinedTextField(
            value = value,
            onValueChange = onValueChange,
            label = { Text(text = label) },
            singleLine = true,
            visualTransformation = if (isVisible) VisualTransformation.None else PasswordVisualTransformation(),
            trailingIcon = {
                TextButton(onClick = { isVisible = !isVisible }) {
                    Text(text = if (isVisible) stringResource(id = R.string.parental_hide_password) else stringResource(id = R.string.parental_show_password))
                }
            },
            keyboardOptions = KeyboardOptions(
                keyboardType = KeyboardType.Password,
                autoCorrect = false
            )
        )
    }
}

@Composable
private fun SupportCard() {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = stringResource(id = R.string.support_section_title),
                style = MaterialTheme.typography.titleMedium
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(text = stringResource(id = R.string.support_description))
            Spacer(modifier = Modifier.height(8.dp))
            Divider()
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = stringResource(id = R.string.support_contact_info),
                style = MaterialTheme.typography.bodySmall
            )
        }
    }
}

@Preview(showBackground = true)
@Composable
private fun PreviewMainScreen() {
    NsfwschutzTheme {
        MainScreen()
    }
}

private suspend fun loadBackendStatus(dispatcher: CoroutineDispatcher = Dispatchers.IO): BackendStatus {
    return withContext(dispatcher) {
        try {
            val url = URL("$BASE_URL/status")
            val connection = (url.openConnection() as HttpURLConnection).apply {
                connectTimeout = 5_000
                readTimeout = 5_000
                requestMethod = "GET"
            }
            connection.connect()
            val responseCode = connection.responseCode
            val stream = if (responseCode in 200..299) connection.inputStream else connection.errorStream
            val message = stream?.use { streamReader ->
                BufferedReader(InputStreamReader(streamReader)).use { it.readText() }
            }.orEmpty()
            connection.disconnect()
            if (responseCode in 200..299) {
                BackendStatus(
                    label = "Server aktiv",
                    message = if (message.isNotBlank()) message else "Verbindung erfolgreich aufgebaut.",
                    timestamp = System.currentTimeMillis(),
                    type = StatusType.SUCCESS
                )
            } else {
                BackendStatus(
                    label = "Warnung",
                    message = "Server meldet Statuscode $responseCode. Bitte prüfen Sie die Warteschlange.",
                    timestamp = System.currentTimeMillis(),
                    type = StatusType.WARNING
                )
            }
        } catch (ex: Exception) {
            BackendStatus(
                label = "Server nicht erreichbar",
                message = "Derzeit gibt es bei uns eine Störung. Bilder können nicht geprüft werden. Wir arbeiten an einer Lösung.",
                timestamp = System.currentTimeMillis(),
                type = StatusType.ERROR
            )
        }
    }
}

private suspend fun loadModels(dispatcher: CoroutineDispatcher = Dispatchers.IO): List<ModelInfo> {
    return withContext(dispatcher) {
        try {
            val url = URL("$BASE_URL/models")
            val connection = (url.openConnection() as HttpURLConnection).apply {
                connectTimeout = 5_000
                readTimeout = 5_000
                requestMethod = "GET"
            }
            connection.connect()
            val responseCode = connection.responseCode
            val stream = if (responseCode in 200..299) connection.inputStream else connection.errorStream
            val body = stream?.use { streamReader ->
                BufferedReader(InputStreamReader(streamReader)).use { it.readText() }
            }.orEmpty()
            connection.disconnect()
            if (body.isBlank()) {
                return@withContext fallbackModels
            }
            val json = JSONArray(body)
            val parsed = mutableListOf<ModelInfo>()
            for (i in 0 until json.length()) {
                val item = json.optJSONObject(i) ?: continue
                parsed.add(item.toModelInfo())
            }
            parsed
        } catch (ex: Exception) {
            fallbackModels
        }
    }
}

private fun JSONObject.toModelInfo(): ModelInfo {
    val slug = optString("slug", optString("id", "model"))
    val display = optString("display", optString("name", slug))
    val version = when {
        has("version") -> optInt("version")
        has("model_version") -> optInt("model_version")
        else -> null
    }
    val tierRaw = optString("tier", when {
        display.contains("fast", ignoreCase = true) -> "fast"
        display.contains("pro", ignoreCase = true) -> "pro"
        display.contains("lite", ignoreCase = true) -> "lite"
        else -> "standard"
    })
    return ModelInfo(
        slug = slug,
        displayName = display,
        version = version,
        tier = tierRaw
    )
}
