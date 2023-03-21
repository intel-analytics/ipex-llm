{{/*
Expand the name of the chart.
*/}}
{{- define "friesian-serving.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "friesian-serving.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/* Fullname suffixed with feature */}}
{{- define "friesian-serving.feature.fullname" -}}
{{- printf "%s-feature" (include "friesian-serving.fullname" .) -}}
{{- end }}

{{/* Fullname suffixed with recall */}}
{{- define "friesian-serving.recall.fullname" -}}
{{- printf "%s-recall" (include "friesian-serving.fullname" .) -}}
{{- end }}

{{/* Fullname suffixed with operator */}}
{{- define "friesian-serving.feature-recall.fullname" -}}
{{- printf "%s-feature-recall" (include "friesian-serving.fullname" .) -}}
{{- end }}

{{/* Fullname suffixed with recall */}}
{{- define "friesian-serving.ranking.fullname" -}}
{{- printf "%s-ranking" (include "friesian-serving.fullname" .) -}}
{{- end }}

{{/* Fullname suffixed with recall */}}
{{- define "friesian-serving.recommender.fullname" -}}
{{- printf "%s-recommender" (include "friesian-serving.fullname" .) -}}
{{- end }}

{{/* redis-server name */}}
{{- define "friesian-serving.redis.fullname" -}}
{{- if .Values.redis.fullnameOverride -}}
{{- .Values.redis.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default "redis" .Values.redis.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Allow the release namespace to be overridden for multi-namespace deployments in combined charts
*/}}
{{- define "friesian-serving.namespace" -}}
  {{- if .Values.namespaceOverride -}}
    {{- .Values.namespaceOverride -}}
  {{- else -}}
    {{- .Release.Namespace -}}
  {{- end -}}
{{- end -}}

{{/*
Define serviceMonitor namespace
*/}}
{{- define "friesian-serving-service-monitor.namespace" -}}
  {{- if .Values.monitorNamespace -}}
    {{- .Values.monitorNamespace -}}
  {{- else -}}
    {{- .Release.Namespace -}}
  {{- end -}}
{{- end -}}


{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "friesian-serving.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "friesian-serving.labels" -}}
helm.sh/chart: {{ include "friesian-serving.chart" . }}
{{ include "friesian-serving.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "friesian-serving.selectorLabels" -}}
app.kubernetes.io/name: {{ include "friesian-serving.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "friesian-serving.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "friesian-serving.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
